import{p as q}from"./chunk-OQCM5LHU-CgAGcoK2.js";import{aq as y,ar as z,as as U,z as j,p as Z,q as H,s as J,g as K,c as Q,b as X,_ as p,l as W,r as Y,d as tt,A as et,E as at,L as rt,W as O,k as nt}from"./index-kwYZwkip.js";import{p as it}from"./gitGraph-YCYPL57B-B7yRvcPn.js";import{o as st}from"./ordinal-Cboi1Yqb.js";import"./_baseUniq-DuOfsaGk.js";import"./_basePickBy-BJEBYxCB.js";import"./clone-CLCg_GYl.js";import"./init-Gi6I4Gst.js";function ot(t,a){return a<t?-1:a>t?1:a>=t?0:NaN}function lt(t){return t}function ct(){var t=lt,a=ot,h=null,o=y(0),g=y(z),x=y(0);function i(e){var r,l=(e=U(e)).length,c,A,m=0,u=new Array(l),n=new Array(l),v=+o.apply(this,arguments),w=Math.min(z,Math.max(-z,g.apply(this,arguments)-v)),f,T=Math.min(Math.abs(w)/l,x.apply(this,arguments)),$=T*(w<0?-1:1),d;for(r=0;r<l;++r)(d=n[u[r]=r]=+t(e[r],r,e))>0&&(m+=d);for(a!=null?u.sort(function(S,C){return a(n[S],n[C])}):h!=null&&u.sort(function(S,C){return h(e[S],e[C])}),r=0,A=m?(w-l*$)/m:0;r<l;++r,v=f)c=u[r],d=n[c],f=v+(d>0?d*A:0)+$,n[c]={data:e[c],index:r,value:d,startAngle:v,endAngle:f,padAngle:T};return n}return i.value=function(e){return arguments.length?(t=typeof e=="function"?e:y(+e),i):t},i.sortValues=function(e){return arguments.length?(a=e,h=null,i):a},i.sort=function(e){return arguments.length?(h=e,a=null,i):h},i.startAngle=function(e){return arguments.length?(o=typeof e=="function"?e:y(+e),i):o},i.endAngle=function(e){return arguments.length?(g=typeof e=="function"?e:y(+e),i):g},i.padAngle=function(e){return arguments.length?(x=typeof e=="function"?e:y(+e),i):x},i}var P=j.pie,F={sections:new Map,showData:!1,config:P},b=F.sections,G=F.showData,ut=structuredClone(P),pt=p(()=>structuredClone(ut),"getConfig"),gt=p(()=>{b=new Map,G=F.showData,Y()},"clear"),dt=p(({label:t,value:a})=>{b.has(t)||(b.set(t,a),W.debug(`added new section: ${t}, with value: ${a}`))},"addSection"),ft=p(()=>b,"getSections"),ht=p(t=>{G=t},"setShowData"),mt=p(()=>G,"getShowData"),R={getConfig:pt,clear:gt,setDiagramTitle:Z,getDiagramTitle:H,setAccTitle:J,getAccTitle:K,setAccDescription:Q,getAccDescription:X,addSection:dt,getSections:ft,setShowData:ht,getShowData:mt},vt=p((t,a)=>{q(t,a),a.setShowData(t.showData),t.sections.map(a.addSection)},"populateDb"),St={parse:p(async t=>{const a=await it("pie",t);W.debug(a),vt(a,R)},"parse")},yt=p(t=>`
  .pieCircle{
    stroke: ${t.pieStrokeColor};
    stroke-width : ${t.pieStrokeWidth};
    opacity : ${t.pieOpacity};
  }
  .pieOuterCircle{
    stroke: ${t.pieOuterStrokeColor};
    stroke-width: ${t.pieOuterStrokeWidth};
    fill: none;
  }
  .pieTitleText {
    text-anchor: middle;
    font-size: ${t.pieTitleTextSize};
    fill: ${t.pieTitleTextColor};
    font-family: ${t.fontFamily};
  }
  .slice {
    font-family: ${t.fontFamily};
    fill: ${t.pieSectionTextColor};
    font-size:${t.pieSectionTextSize};
    // fill: white;
  }
  .legend text {
    fill: ${t.pieLegendTextColor};
    font-family: ${t.fontFamily};
    font-size: ${t.pieLegendTextSize};
  }
`,"getStyles"),xt=yt,At=p(t=>{const a=[...t.entries()].map(o=>({label:o[0],value:o[1]})).sort((o,g)=>g.value-o.value);return ct().value(o=>o.value)(a)},"createPieArcs"),wt=p((t,a,h,o)=>{W.debug(`rendering pie chart
`+t);const g=o.db,x=tt(),i=et(g.getConfig(),x.pie),e=40,r=18,l=4,c=450,A=c,m=at(a),u=m.append("g");u.attr("transform","translate("+A/2+","+c/2+")");const{themeVariables:n}=x;let[v]=rt(n.pieOuterStrokeWidth);v??(v=2);const w=i.textPosition,f=Math.min(A,c)/2-e,T=O().innerRadius(0).outerRadius(f),$=O().innerRadius(f*w).outerRadius(f*w);u.append("circle").attr("cx",0).attr("cy",0).attr("r",f+v/2).attr("class","pieOuterCircle");const d=g.getSections(),S=At(d),C=[n.pie1,n.pie2,n.pie3,n.pie4,n.pie5,n.pie6,n.pie7,n.pie8,n.pie9,n.pie10,n.pie11,n.pie12],D=st(C);u.selectAll("mySlices").data(S).enter().append("path").attr("d",T).attr("fill",s=>D(s.data.label)).attr("class","pieCircle");let L=0;d.forEach(s=>{L+=s}),u.selectAll("mySlices").data(S).enter().append("text").text(s=>(s.data.value/L*100).toFixed(0)+"%").attr("transform",s=>"translate("+$.centroid(s)+")").style("text-anchor","middle").attr("class","slice"),u.append("text").text(g.getDiagramTitle()).attr("x",0).attr("y",-(c-50)/2).attr("class","pieTitleText");const M=u.selectAll(".legend").data(D.domain()).enter().append("g").attr("class","legend").attr("transform",(s,k)=>{const E=r+l,_=E*D.domain().length/2,B=12*r,V=k*E-_;return"translate("+B+","+V+")"});M.append("rect").attr("width",r).attr("height",r).style("fill",D).style("stroke",D),M.data(S).append("text").attr("x",r+l).attr("y",r-l).text(s=>{const{label:k,value:E}=s.data;return g.getShowData()?`${k} [${E}]`:k});const I=Math.max(...M.selectAll("text").nodes().map(s=>(s==null?void 0:s.getBoundingClientRect().width)??0)),N=A+e+r+l+I;m.attr("viewBox",`0 0 ${N} ${c}`),nt(m,c,N,i.useMaxWidth)},"draw"),Ct={draw:wt},Wt={parser:St,db:R,renderer:Ct,styles:xt};export{Wt as diagram};
