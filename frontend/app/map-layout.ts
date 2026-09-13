export function spreadMapMarkers(points: { x: number; y: number; radius: number }[]) {
  const spread = points.map(point => ({ ...point }));
  // Pixel offsets keep nearby buttons operable; leader lines retain exact station coordinates.
  for (let pass = 0; pass < 40; pass++) {
    for (let i = 0; i < spread.length; i++) {
      for (let j = i + 1; j < spread.length; j++) {
        let dx = spread[j].x - spread[i].x; let dy = spread[j].y - spread[i].y;
        let distance = Math.hypot(dx, dy); const minimum = spread[i].radius + spread[j].radius + 7;
        if (distance >= minimum) continue;
        if (distance < 0.01) { dx = Math.cos(i + j); dy = Math.sin(i + j); distance = 1; }
        const shift = (minimum - distance) / 2;
        spread[i].x -= dx / distance * shift; spread[i].y -= dy / distance * shift;
        spread[j].x += dx / distance * shift; spread[j].y += dy / distance * shift;
      }
    }
  }
  return spread;
}
