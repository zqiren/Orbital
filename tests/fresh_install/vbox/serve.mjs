// Loopback-only static file server for prepare-vm.ps1: the guest fetches
// guest-firstlogon.ps1, the rig's public key, tester's password and the
// OpenSSH MSI from http://10.0.2.2:<port> (VirtualBox NAT's alias for the
// host's 127.0.0.1, reachable only while nat-localhostreachable is on).
// Bound to 127.0.0.1, so nothing is exposed to the LAN.
//   node serve.mjs <dir> <port>
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';

const [dir, port] = [path.resolve(process.argv[2]), Number(process.argv[3] || 8399)];
const files = new Set(fs.readdirSync(dir));
http
  .createServer((req, res) => {
    const name = decodeURIComponent((req.url || '/').split('?')[0].replace(/^\/+/, ''));
    console.log(`${new Date().toISOString()} ${req.method} /${name}`);
    if (!files.has(name)) {
      res.writeHead(404).end();
      return;
    }
    const f = path.join(dir, name);
    res.writeHead(200, { 'content-type': 'application/octet-stream', 'content-length': fs.statSync(f).size });
    fs.createReadStream(f).pipe(res);
  })
  .listen(port, '127.0.0.1', () => console.log(`serving ${dir} on 127.0.0.1:${port}`));
