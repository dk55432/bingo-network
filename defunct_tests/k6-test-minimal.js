import ws from 'k6/ws';
import { check, sleep } from 'k6';

export const options = { vus: 1, duration: '20s' };
const WS_URL = 'ws://10.0.0.153:8000/ws';

export default function () {
  ws.connect(WS_URL, {}, (socket) => {
    let saw = false;

    socket.on('open', () => console.log('open'));
    socket.on('close', () => console.log('close'));
    socket.on('error', (e) => console.log('error:', e));

    socket.send(JSON.stringify({ type: 'ping' }));

    const start = Date.now();
    while (Date.now() - start < 10000 && !saw) {
      try {
        const msg = socket.recv(); // <-- key change
        if (msg !== undefined && msg !== null) {
          console.log('recv returned:', msg);
          saw = true;
        }
      } catch (e) {
        // ignore and keep waiting
      }
      sleep(0.1);
    }

    check(saw, { 'received message via recv': (v) => v });
    socket.close();
  });
}
