
    (() => {
      const root = document.getElementById('camera-expansion-plan');
      const ns = 'http://www.w3.org/2000/svg';
      const points = [
        {id:'G-01',x:24,y:45,p:'must',name:'西侧出入口断面',cfg:'每方向 1 台车牌抓拍 + 1 台车辆全景，用于首次建档和普通/货车分流。'},
        {id:'G-02',x:87,y:44,p:'must',name:'东北侧出入口断面',cfg:'在车辆进入内部道路前完成车牌、车型、颜色和 ReID 特征采集。'},
        {id:'G-03',x:43,y:94,p:'must',name:'南侧行政通道候选入口',cfg:'优先现场确认是否为行政通道；确认后按进、出方向分别设置近景与全景。'},
        {id:'J-01',x:52,y:30,p:'must',name:'北区首个分叉口',cfg:'覆盖所有转向，避免车辆进入北侧支路后失联；建议每个进入方向至少有一个清晰正/后视角。'},
        {id:'J-02',x:44,y:53,p:'must',name:'中央汇合与分流点',cfg:'这是跨摄像头轨迹关联的核心节点，应覆盖西侧、北侧、东侧和南侧来车。'},
        {id:'J-03',x:72,y:53,p:'next',name:'东侧道路转向点',cfg:'补充侧面车型与外观特征，减少东侧长路段的轨迹歧义。'},
        {id:'J-04',x:48,y:72,p:'must',name:'南区内部路口',cfg:'连接南侧入口和园区内部道路，用于确认车辆进入、离开或折返。'},
        {id:'R-01',x:66,y:91,p:'next',name:'东南长路段中继点',cfg:'使用视频跟踪摄像机补足长距离空窗，不承担主要车牌识别。'}
      ];
      const group = root.querySelector('#proposal-points');
      const title = root.querySelector('#proposal-title');
      const desc = root.querySelector('#proposal-desc');
      points.forEach(p => {
        const g = document.createElementNS(ns,'g');
        g.setAttribute('class',`point ${p.p}`);
        g.setAttribute('role','button');
        g.setAttribute('aria-label',`${p.id} ${p.name}`);
        g.innerHTML = `<circle cx="${p.x}" cy="${p.y}" r="1.18"></circle><text x="${p.x+1.7}" y="${p.y+.75}">${p.id}</text>`;
        g.addEventListener('click',() => {
          title.textContent = `${p.id} · ${p.name}`;
          desc.textContent = p.cfg;
          window.openai?.setWidgetState?.({modelContent:{selectedPoint:p.id,location:p.name},privateContent:null}).catch(()=>{});
        });
        group.appendChild(g);
      });
    })();
  