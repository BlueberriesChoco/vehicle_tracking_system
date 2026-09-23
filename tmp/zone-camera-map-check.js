
    (() => {
      const root = document.getElementById('zone-camera-map');
      const ns = 'http://www.w3.org/2000/svg';
      const cameras = [
        ['B-01','B',75.1,7.9,'192.178.2.50'],['V-02','V',79.0,8.2,'192.178.2.3'],
        ['B-02','B',63.5,13.7,'192.178.2.51'],['V-22','V',64.6,20.5,'192.178.2.23'],
        ['V-21','V',62.2,25.1,'192.178.2.22'],['V-03','V',67.7,26.1,'192.178.2.4'],
        ['B-04','B',47.7,32.3,'192.178.2.53'],['V-20','V',54.0,34.5,'192.178.2.21'],
        ['V-04','V',62.5,34.6,'192.178.2.5'],['B-05','B',45.9,37.3,'192.178.2.54'],
        ['B-03','B',61.7,38.0,'192.178.2.52'],['V-19','V',49.0,42.4,'192.178.2.20'],
        ['V-06','V',53.0,41.0,'192.178.2.7'],['V-05','V',55.4,43.0,'192.178.2.6'],
        ['B-06','B',43.7,44.1,'192.178.2.55'],['V-17','V',48.0,48.3,'192.178.2.18'],
        ['V-18','V',43.0,50.8,'192.178.2.19'],['V-07','V',51.2,51.8,'192.178.2.8'],
        ['V-16','V',37.5,56.2,'192.178.2.17'],['V-08','V',43.0,58.2,'192.178.2.9'],
        ['V-23','V',46.5,59.3,'192.178.2.24'],['B-07','B',48.1,61.7,'192.178.2.56'],
        ['B-08','B',46.8,63.2,'192.178.2.57'],['B-09','B',48.5,64.4,'192.178.2.58'],
        ['V-15','V',33.7,63.8,'192.178.2.16'],['B-11','B',21.5,66.0,'192.178.2.60'],
        ['V-13','V',22.8,74.1,'192.178.2.14'],['V-14','V',21.6,77.2,'192.178.2.15'],
        ['V-11','V',28.4,77.0,'192.178.2.12'],['V-10','V',27.7,79.5,'192.178.2.11'],
        ['V-12','V',24.1,84.0,'192.178.2.13'],['V-09','V',39.5,74.3,'192.178.2.10'],
        ['V-01','V',41.1,76.2,'192.178.2.1'],['B-10','B',44.7,76.5,'192.178.2.59']
      ];
      const group = root.querySelector('#camera-points');
      const title = root.querySelector('#selected-title');
      const desc = root.querySelector('#selected-desc');
      cameras.forEach(([id,type,x,y,ip]) => {
        const g = document.createElementNS(ns,'g');
        g.setAttribute('class',`camera ${type.toLowerCase()}`);
        g.setAttribute('role','button');
        g.setAttribute('aria-label',`${id}，${type === 'B' ? '路口抓拍摄像机' : '视频跟踪摄像机'}，${ip}`);
        g.innerHTML = `<circle cx="${x}" cy="${y}" r="0.78"></circle><text x="${x+1.15}" y="${y+0.65}">${id}</text>`;
        g.addEventListener('click',() => {
          title.textContent = `${id} · ${type === 'B' ? '路口抓拍摄像机' : '视频跟踪摄像机'}`;
          desc.textContent = `图纸地址：${ip}；位置为竣工图相对坐标，需现场或 GIS 坐标复核。`;
        });
        group.appendChild(g);
      });
      root.querySelectorAll('.fence-zone').forEach(zone => zone.addEventListener('click',() => {
        title.textContent = zone.dataset.name;
        desc.textContent = `${zone.dataset.desc}；围网图中可见连续摄像机符号，但暂无可核验的设备编号/IP。`;
      }));
      root.querySelectorAll('[role="tab"]').forEach(tab => tab.addEventListener('click',() => {
        root.querySelectorAll('[role="tab"]').forEach(t => { t.classList.toggle('active',t===tab); t.setAttribute('aria-selected',String(t===tab)); });
        root.querySelector('#camera-panel').hidden = tab.id !== 'camera-tab';
        root.querySelector('#fence-panel').hidden = tab.id !== 'fence-tab';
      }));
    })();
  