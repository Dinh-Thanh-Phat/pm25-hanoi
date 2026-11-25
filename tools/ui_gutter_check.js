// Kiểm tra lề trái/phải ở nhiều khổ màn hình. Chạy bằng Playwright (browser_run_code) sau MỖI lần sửa UI.
// Trả về {khổ: {exp, bad[]}}; bad phải rỗng (bỏ qua chính các khung .wrap vì hộp của chúng gồm cả lề).
async (page) => {
  const res = {};
  for (const w of [320, 360, 390, 430, 600, 768, 820, 1024, 1180, 1280, 1440, 1680, 1866, 1920, 2560]) {
    await page.setViewportSize({width: w, height: 900});
    await page.goto('http://localhost:8200/?nointro&t=' + Date.now());
    await page.waitForFunction(() => !document.body.classList.contains('wait'));
    await page.waitForTimeout(300);
    res[w] = await page.evaluate(() => {
      const cw = document.documentElement.clientWidth, gut = parseFloat(getComputedStyle(document.querySelector('.wrap')).paddingLeft);
      const mw = parseFloat(getComputedStyle(document.querySelector('.wrap')).maxWidth), exp = Math.max(0, (cw - mw) / 2) + gut, bad = new Set();
      document.querySelectorAll('.hero-grid *, .top *, main *, footer *, .photo *').forEach(e => {
        if (e.classList.contains('wrap') || e.closest('.haze, .leaflet-container, .tbl, svg, details:not([open]) > :not(summary)')) return;
        const b = e.getBoundingClientRect(); if (!b.width || !b.height) return;
        if (b.left < exp - 1 || b.right > cw - exp + 1) bad.add(`${e.tagName}${e.id ? '#' + e.id : ''}.${String(e.className).split(' ')[0]} ${Math.round(b.left)}..${Math.round(cw - b.right)}`);
      });
      return {exp: Math.round(exp), ov: document.documentElement.scrollWidth - innerWidth, bad: [...bad].slice(0, 8)};
    });
  }
  return res;
}
