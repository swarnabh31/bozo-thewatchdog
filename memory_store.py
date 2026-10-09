"""BoZo local, inspectable, deletable SQLite memory. No models or network calls."""
from __future__ import annotations
import json, os, re, sqlite3, threading, time
DEFAULT_DB = os.path.join(os.environ.get('APPDATA', os.path.expanduser('~')), 'screen_copilot', 'memory.db')
VALID_CATEGORIES = {'error','tip','chat_me','chat_bozo','note'}
RETENTION_DAYS = {'error':90,'tip':90,'note':90,'chat_me':30,'chat_bozo':30}
_TOKEN = re.compile(r'[A-Za-z0-9_./:#+-]{3,}')
def _tokenize(text): return [x.lower() for x in _TOKEN.findall(text or '')]
def jaccard_overlap(a,b):
    aa,bb=set(_tokenize(a)),set(_tokenize(b)); union=aa|bb
    return len(aa&bb)/len(union) if union else 0.0
class MemoryStore:
    """Thread-safe SQLite memory store used by the BoZo GUI."""
    def __init__(self,path=DEFAULT_DB):
        self.path=path; os.makedirs(os.path.dirname(path) or '.',exist_ok=True); self._lock=threading.RLock()
        self._conn=sqlite3.connect(path,check_same_thread=False); self._conn.execute('PRAGMA journal_mode=WAL'); self._conn.execute('PRAGMA foreign_keys=ON')
        self._conn.executescript('''CREATE TABLE IF NOT EXISTS memories(id INTEGER PRIMARY KEY,created_at REAL NOT NULL,category TEXT NOT NULL,window_title TEXT DEFAULT '',sig TEXT DEFAULT '',text TEXT NOT NULL,helpful INTEGER); CREATE INDEX IF NOT EXISTS idx_mem_created ON memories(created_at); CREATE INDEX IF NOT EXISTS idx_mem_sig ON memories(sig); CREATE INDEX IF NOT EXISTS idx_mem_cat ON memories(category); CREATE TABLE IF NOT EXISTS feedback(id INTEGER PRIMARY KEY,created_at REAL NOT NULL,sig TEXT NOT NULL,verdict INTEGER NOT NULL,note TEXT DEFAULT ''); CREATE INDEX IF NOT EXISTS idx_fb_sig ON feedback(sig);'''); self._conn.commit(); self._fts=False
    def add(self,category,text,window_title='',sig='',helpful=None):
        category=category if category in VALID_CATEGORIES else 'note'; text=(text or '').strip()
        if not text:return -1
        with self._lock:
            c=self._conn.execute('INSERT INTO memories(created_at,category,window_title,sig,text,helpful) VALUES(?,?,?,?,?,?)',(time.time(),category,window_title or '',sig or '',text,helpful)); self._conn.commit(); return int(c.lastrowid)
    def record_chat(self,role,text,window_title=''):
        return self.add({'me':'chat_me','bozo':'chat_bozo','assistant':'chat_bozo'}.get(role,'note'),text,window_title)
    def record_error(self,sig,text,window_title=''): return self.add('error',text,window_title,sig)
    def record_tip(self,sig,text,window_title=''): return self.add('tip',text,window_title,sig)
    def record_feedback(self,sig,verdict,note=''):
        if not sig:return
        with self._lock:self._conn.execute('INSERT INTO feedback(created_at,sig,verdict,note) VALUES(?,?,?,?)',(time.time(),sig,1 if verdict else 0,note or '')); self._conn.commit()
    def set_helpful(self,memory_id,verdict):
        with self._lock:self._conn.execute('UPDATE memories SET helpful=? WHERE id=?',(1 if verdict else 0,memory_id)); self._conn.commit()
    def recall(self,message='',sigs=None,window_title='',k=6,categories=None):
        sigs=set(s for s in (sigs or []) if s); now=time.time(); half=30*86400
        with self._lock:
            if categories:
                ph=','.join('?'*len(categories)); rows=self._conn.execute(f'SELECT id,created_at,category,window_title,sig,text FROM memories WHERE category IN ({ph}) ORDER BY created_at DESC LIMIT 400',tuple(categories)).fetchall()
            else: rows=self._conn.execute('SELECT id,created_at,category,window_title,sig,text FROM memories ORDER BY created_at DESC LIMIT 400').fetchall()
        out=[]
        for rid,ts,cat,win,sig,text in rows:
            score=10 if sig and sig in sigs else 0; why=['same error seen before'] if score else []
            overlap=jaccard_overlap(message,text)
            if overlap>.05:score+=6*overlap; why.append(f'keyword overlap {overlap:.0%}')
            if window_title and jaccard_overlap(window_title,win)>.15:score+=1.5; why.append('same window')
            if cat in ('error','tip') and score:score*=1.25
            score*=.5+0.5**(max(0,now-ts)/half)
            if score>.15:out.append({'id':rid,'category':cat,'text':text,'window_title':win,'sig':sig,'created_at':ts,'score':round(score,3),'why':', '.join(why) or 'recency'})
        return sorted(out,key=lambda x:x['score'],reverse=True)[:max(0,k)]
    def format_context(self,results,budget=2200):
        if not results:return ''
        lines=['Relevant things BoZo has seen before (local memory):']; used=len(lines[0])
        for r in results:
            when=time.strftime('%b %d %H:%M',time.localtime(r['created_at'])); line=f"- [{r['category']}] ({when}) {r['text'].strip()}"+(f"  <-- {r['why']}" if r.get('why') else '')
            if used+len(line)>budget:break
            lines.append(line); used+=len(line)
        return '\n'.join(lines)
    def recent_chat(self,n=10):
        with self._lock: rows=self._conn.execute("SELECT category,text,created_at FROM memories WHERE category IN ('chat_me','chat_bozo') ORDER BY created_at DESC LIMIT ?",(n,)).fetchall()
        return [{'role':'me' if c=='chat_me' else 'bozo','text':t,'created_at':ts} for c,t,ts in reversed(rows)]
    def stats(self):
        with self._lock:
            cats=self._conn.execute('SELECT category,COUNT(*) FROM memories GROUP BY category').fetchall(); total=self._conn.execute('SELECT COUNT(*) FROM memories').fetchone()[0]; earliest=self._conn.execute('SELECT MIN(created_at) FROM memories').fetchone()[0]; latest=self._conn.execute('SELECT MAX(created_at) FROM memories').fetchone()[0]; fb=self._conn.execute('SELECT verdict,COUNT(*) FROM feedback GROUP BY verdict').fetchall()
        fbd=dict(fb or []); return {'total_memories':total,'by_category':dict(cats),'earliest':earliest,'latest':latest,'feedback_helpful':fbd.get(1,0),'feedback_not_yet':fbd.get(0,0),'db_bytes':os.path.getsize(self.path) if os.path.exists(self.path) else 0,'fts5':False}
    def history_for_sig(self,sig):
        with self._lock:
            mc=self._conn.execute('SELECT COUNT(*) FROM memories WHERE sig=?',(sig,)).fetchone()[0]; fb=self._conn.execute('SELECT verdict,COUNT(*) FROM feedback WHERE sig=? GROUP BY verdict',(sig,)).fetchall(); last=self._conn.execute('SELECT MAX(created_at) FROM feedback WHERE sig=?',(sig,)).fetchone()[0]
        d=dict(fb or []); return {'seen':mc+d.get(1,0)+d.get(0,0),'helpful':d.get(1,0),'not_yet':d.get(0,0),'last_seen':last}
    def delete_ids(self,ids):
        if not ids:return 0
        with self._lock:c=self._conn.execute('DELETE FROM memories WHERE id IN ('+','.join('?'*len(ids))+')',list(ids)); self._conn.commit(); return c.rowcount
    def delete_category(self,category):
        with self._lock:c=self._conn.execute('DELETE FROM memories WHERE category=?',(category,)); self._conn.commit(); return c.rowcount
    def delete_all(self):
        with self._lock:a=self._conn.execute('DELETE FROM memories'); b=self._conn.execute('DELETE FROM feedback'); self._conn.commit(); return a.rowcount+b.rowcount
    def prune(self,days_by_category=None):
        days_by_category=dict(days_by_category or RETENTION_DAYS); removed=0
        with self._lock:
            for cat,days in days_by_category.items(): removed+=self._conn.execute('DELETE FROM memories WHERE category=? AND created_at<?',(cat,time.time()-int(days)*86400)).rowcount
            removed+=self._conn.execute('DELETE FROM feedback WHERE created_at<?',(time.time()-int(days_by_category.get('error',90))*86400,)).rowcount; self._conn.commit()
        return removed
    def export_json(self,path):
        with self._lock: rows=self._conn.execute('SELECT id,created_at,category,window_title,sig,text,helpful FROM memories ORDER BY created_at DESC').fetchall()
        data=[dict(zip(('id','created_at','category','window_title','sig','text','helpful'),r)) for r in rows]
        with open(path,'w',encoding='utf-8') as f:json.dump({'bozo_memory_export':True,'exported_at':time.time(),'memories':data},f,ensure_ascii=False,indent=2)
        return len(data)
    def list_memories(self,category=None,limit=800):
        cats=list(category) if isinstance(category,(list,tuple,set)) else ([category] if category else [])
        with self._lock:
            if cats: rows=self._conn.execute('SELECT id,created_at,category,window_title,text FROM memories WHERE category IN ('+','.join('?'*len(cats))+') ORDER BY created_at DESC LIMIT ?',[*cats,limit]).fetchall()
            else: rows=self._conn.execute('SELECT id,created_at,category,window_title,text FROM memories ORDER BY created_at DESC LIMIT ?',(limit,)).fetchall()
        return [dict(zip(('id','created_at','category','window_title','text'),r)) for r in rows]
    def close(self):
        with self._lock:self._conn.commit(); self._conn.close()
