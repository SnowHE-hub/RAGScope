// Optional local visual QA. Uses the existing bundled sharp; never a browser.
// node preview_diagrams.js <absolute path to existing sharp package>
const fs = require('fs');
const path = require('path');
const sharp = require(process.argv[2] || 'sharp');
const source = path.join(__dirname, 'diagrams');
const output = path.join(__dirname, 'previews');
fs.mkdirSync(output, { recursive: true });
(async () => {
  const rows = [];
  for (const name of fs.readdirSync(source).filter(x => x.endsWith('.svg')).sort()) {
    const destination = path.join(output, name.replace(/\.svg$/, '.png'));
    const result = await sharp(path.join(source, name), { density: 120 }).png().toFile(destination);
    rows.push({ source: name, output: destination, width: result.width, height: result.height, bytes: result.size });
  }
  console.log(JSON.stringify(rows, null, 2));
})().catch(error => { console.error(error); process.exitCode = 1; });
