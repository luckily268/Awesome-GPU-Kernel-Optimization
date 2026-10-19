(function(root){
'use strict';
const categories={cuda:{name:'CUDA',full:'CUDA 生态',en:'CUDA-centric',icon:'⌘',color:'#33765b'},dsl:{name:'高级 DSL',full:'高级内核 DSL',en:'High-level kernel DSL',icon:'⌁',color:'#6870a7'},alt:{name:'替代加速器',full:'替代加速器',en:'Alternative accelerators',icon:'◇',color:'#ad7a46'},cross:{name:'跨平台',full:'跨平台与多后端',en:'Cross-platform & multi-backend',icon:'⇄',color:'#447f99'},unseen:{name:'未知 ISA',full:'未知架构与 ISA',en:'Unseen architectures & ISAs',icon:'✳',color:'#967397'}};
const controllers={single:'Single-Agent',multi:'Multi-Agent',training:'Training-based',unknown:'待确认'};
const roles={method:'方法研究',benchmark:'Benchmark'};
const evidences={primary:'已读原文首页 / 摘要',survey:'依据综述整理'};
const emptyState=()=>({category:null,controllers:[],years:[],roles:[],evidence:[],query:'',sort:'latest',view:'map',month:null});
function filter(items,s,ignore){return items.filter(x=>(ignore==='category'||!s.category||s.category===x.category)&&(ignore==='controllers'||!s.controllers.length||s.controllers.some(t=>x.controllers.includes(t)))&&(ignore==='years'||!s.years.length||s.years.includes(String(x.year||'unknown')))&&(ignore==='roles'||!s.roles.length||s.roles.some(t=>x.roles.includes(t)))&&(ignore==='evidence'||!s.evidence.length||s.evidence.includes(x.evidence))&&(!s.month||x.date?.slice(0,7)===s.month)&&(!s.query.trim()||s.query.toLowerCase().trim().split(/\s+/).every(q=>[x.name,x.citation,x.artifact,x.summary,x.rationale,...x.benchmarks,...x.controllers.map(t=>controllers[t]),categories[x.category].full].join(' ').toLowerCase().includes(q))));}
function sort(items,mode){return [...items].sort((a,b)=>{if(mode==='name')return a.name.localeCompare(b.name,'en');const ad=a.year?(a.year*100+(a.month||0)):0,bd=b.year?(b.year*100+(b.month||0)):0;if(!ad||!bd)return !ad&&!bd?a.name.localeCompare(b.name):!ad?1:-1;return (mode==='earliest'?ad-bd:bd-ad)||a.name.localeCompare(b.name,'en');});}
function dateLabel(x){if(!x.year)return '年份待核实';if(!x.month)return x.year+' · 月份待核实';return x.year+'.'+String(x.month).padStart(2,'0');}
function isPending(x){return !x.month||x.controllers.includes('unknown')||x.notes.length>0;}
function timeline(items){const buckets={};const unknown=[];for(const x of items){if(!x.year||!x.month){unknown.push(x);continue;}const key=x.year+'-'+String(x.month).padStart(2,'0');(buckets[key]??=[]).push(x);}return {buckets,unknown};}
const api={categories,controllers,roles,evidences,emptyState,filter,sort,dateLabel,isPending,timeline};root.AtlasCore=api;if(typeof module!=='undefined')module.exports=api;
})(typeof globalThis!=='undefined'?globalThis:this);
