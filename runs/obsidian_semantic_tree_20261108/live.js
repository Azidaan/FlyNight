(async()=>{
const fs=require("fs"),root=app.vault.adapter.basePath;
const m=JSON.parse(fs.readFileSync(root+"/.graph-visible.json","utf8")),actual=new Set(app.vault.getMarkdownFiles().map(f=>f.path));
const missing=[],missingCache=[],edges=[],unresolved=[],wrong=[],emptyLeaves=[];
const children=new Set(Object.values(m.nodes).map(n=>n.parent));
for(const [id,n] of Object.entries(m.nodes)){
 if(!actual.has(n.path)){missing.push(n.path);continue}
 const c=app.metadataCache.getFileCache(app.vault.getAbstractFileByPath(n.path));
 if(!c)missingCache.push(n.path);
 for(const [p,count] of Object.entries(app.metadataCache.resolvedLinks[n.path]||{}))edges.push([n.path,p,count]);
 for(const [p,count] of Object.entries(app.metadataCache.unresolvedLinks[n.path]||{}))unresolved.push([n.path,p,count]);
 if(n.parent){const p=m.nodes[n.parent].path;if(!app.metadataCache.resolvedLinks[p]?.[n.path])wrong.push([p,n.path]);}
 if(!children.has(id)){
  const text=fs.readFileSync(root+"/"+n.path,"utf8").replace(/^---[\s\S]*?---/,"").replace(/^#+.*$/gm,"").replace(/\[[^\]]*\]\([^)]*\)/g,"").trim();
  if(!text||/^\*\*Статус:\*\*[^\n]*$/.test(text))emptyLeaves.push(n.path);
 }
}
const r={version:"1.14.5",nodes:Object.keys(m.nodes).length,indexMissing:missing,cacheMissing:missingCache,missingEdges:wrong,edges,unresolved,emptyLeaves};
fs.writeFileSync("E:/Dissertation/runs/obsidian_semantic_tree_20261108/"+(globalThis.integrityPhase||"after_reload")+".json",JSON.stringify(r,null,2));
return JSON.stringify({nodes:r.nodes,indexMissing:missing.length,cacheMissing:missingCache.length,missingEdges:wrong.length,unresolved:unresolved.length,emptyLeaves:emptyLeaves.length,edges:edges.length});
})()
