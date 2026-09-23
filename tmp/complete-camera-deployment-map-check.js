
    (() => {
      const root = document.getElementById('complete-camera-deployment-map');
      const ns = 'http://www.w3.org/2000/svg';
      const points = [
        {id:'G-01',type:'gate',x:24,y:45,name:'西侧边界出入口',ref:'西侧围网与园区横向道路交叉处，紧邻图中西侧红色通行箭头。',views:'建议 4 个视角：进、出方向各 1 个车牌近景和 1 个车辆全景。',purpose:'入口建档；车牌 ROI 调试到触发线处不低于 100 px；普通车辆与货车首次分流。'},
        {id:'G-02',type:'gate',x:87,y:44,name:'东北侧边界出入口',ref:'东北折角东侧，围网与东侧道路连接处，图中右侧红色箭头位置。',views:'建议 4 个视角：双向车牌近景 + 双向全景分类。',purpose:'东北出入口建档、离场闭环和异常折返确认。'},
        {id:'G-03',type:'gate',x:43,y:94,name:'南侧行政通道候选入口',ref:'南侧围网凹口处，中央南北道路延伸至南侧边界的位置。',views:'确认行政通道后建议 4 个视角；若仅单向通行可减为 2 个视角。',purpose:'行政车辆研究的主入口候选点；施工前必须现场核实通道属性和车流方向。'},
        {id:'J-01',type:'junction',x:52,y:30,name:'北区西侧首个分叉',ref:'北部区域西侧折角内，南北道路与北侧横向道路交汇处。',views:'建议 3 个全景跟踪视角，分别覆盖三个来车方向。',purpose:'防止车辆进入北侧支路后断轨，并输出转向标签。'},
        {id:'J-02',type:'junction',x:44,y:53,name:'中央西侧汇合点',ref:'西侧入口向园区内部延伸后的第一个主要汇合点。',views:'建议 4 个全景视角；优先复用附近现有 V 类视频跟踪摄像机。',purpose:'连接西门、北区和南区轨迹，是跨镜关联的核心节点。'},
        {id:'J-03',type:'junction',x:72,y:53,name:'东侧中部三岔口',ref:'园区东侧中部，内部横向道路接入东侧纵向道路的位置。',views:'建议 3 个全景视角，覆盖直行及两侧转向。',purpose:'补充车辆侧面、颜色和车型特征，降低同色车辆混淆。'},
        {id:'J-04',type:'junction',x:48,y:72,name:'南区中央分叉',ref:'中央南北道路中段，与南区横向支路交汇的位置。',views:'建议 3 个全景视角。',purpose:'确认车辆向南门、东南区或西南区的去向。'},
        {id:'J-05',type:'junction',x:84,y:69,name:'东侧道路折转点',ref:'东侧围网中下段，内部道路贴近东侧边界转弯的位置。',views:'建议 2 个相向全景视角。',purpose:'解决急弯遮挡和车辆短时重叠造成的 ID 切换。'},
        {id:'J-06',type:'junction',x:35,y:82,name:'西南片区转向点',ref:'西南片区道路与通往南侧入口道路的连接处。',views:'建议 2 至 3 个全景视角，按现场支路数量确定。',purpose:'覆盖西南片区进出路径和折返车辆。'},
        {id:'J-07',type:'junction',x:69,y:31,name:'北区内部主路口',ref:'北部园区内部横向主路与纵向支路交汇位置。',views:'建议 3 个全景视角。',purpose:'补齐北部内部道路，不让车辆仅在出入口出现、进入园区后失联。'},
        {id:'R-01',type:'relay',x:65,y:91,name:'东南长路段中继',ref:'东南侧长直道路中段，位于东侧折转点与南侧入口之间。',views:'建议 1 台高位全景跟踪摄像机。',purpose:'缩短跨镜时间间隔；不承担主车牌识别任务。'},
        {id:'R-02',type:'relay',x:34,y:36,name:'西北入口连接路中继',ref:'西侧入口至北区首个分叉之间的斜向连接道路中段。',views:'建议 1 台顺行方向全景摄像机。',purpose:'补偿西北连接道路较长、弯道遮挡造成的轨迹空窗。'},
        {id:'R-03',type:'relay',x:47,y:63,name:'中央南北道路中继',ref:'中央汇合点与南区分叉之间的南北向道路中段。',views:'建议 1 台全景跟踪摄像机；如现有 V 点覆盖清晰则直接复用。',purpose:'保持中央主通道 ReID 特征连续。'},
        {id:'R-04',type:'relay',x:78,y:57,name:'东侧纵向道路中继',ref:'东侧中部三岔口与东侧折转点之间的纵向道路中段。',views:'建议 1 台全景跟踪摄像机。',purpose:'减少东侧长路段车辆并行或超车时的关联歧义。'}
      ];
      const group = root.querySelector('#deployment-points');
      const select = root.querySelector('#deployment-select');
      const title = root.querySelector('#deployment-title');
      const location = root.querySelector('#deployment-location');
      const config = root.querySelector('#deployment-config');
      const purpose = root.querySelector('#deployment-purpose');
      const nodes = new Map();
      const typeName = {gate:'出入口识别组',junction:'路口跟踪组',relay:'路段中继组'};
      function choose(id) {
        const p = points.find(item => item.id === id);
        nodes.forEach((node,key) => node.classList.toggle('selected',key === id));
        select.value = id;
        title.textContent = `${p.id} · ${p.name} · ${typeName[p.type]} · 图纸坐标 X=${p.x}%, Y=${p.y}%`;
        location.textContent = `位置：${p.ref}`;
        config.textContent = `配置：${p.views}`;
        purpose.textContent = `作用：${p.purpose}`;
        window.openai?.setWidgetState?.({modelContent:{selectedPoint:p.id,location:p.name,planCoordinate:{x:p.x,y:p.y}},privateContent:null}).catch(()=>{});
      }
      points.forEach(p => {
        const option = document.createElement('option');
        option.value = p.id;
        option.textContent = `${p.id} · ${p.name}`;
        select.appendChild(option);
        const g = document.createElementNS(ns,'g');
        g.setAttribute('class',`point ${p.type}`);
        g.setAttribute('role','button');
        g.setAttribute('aria-label',`${p.id} ${p.name}`);
        g.innerHTML = `<circle cx="${p.x}" cy="${p.y}" r="1.12"></circle><text x="${p.x+1.65}" y="${p.y+.72}">${p.id}</text>`;
        g.addEventListener('click',() => choose(p.id));
        group.appendChild(g);
        nodes.set(p.id,g);
      });
      select.addEventListener('change',() => choose(select.value));
      choose('G-03');
    })();
  