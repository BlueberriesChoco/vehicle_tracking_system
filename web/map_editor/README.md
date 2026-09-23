# Zone Map Editor

Open `index.html` directly in a modern browser. No package installation or web server is required.

The editor uses `assets/zone-a-base-map.jpg` as the base map and supports:

- enterprise areas;
- drag-resizable enterprise rectangles;
- named road polylines;
- camera points and metadata;
- camera details shown on pointer hover;
- layer visibility;
- drag editing;
- undo and redo;
- browser-local automatic saving;
- JSON import and export;
- PNG export.

Map coordinates use the source image coordinate system (`1280 x 1809`). They are not GIS longitude and latitude. Exported JSON can later be converted to `config/site/*.yaml` after real-world coordinates are available.
