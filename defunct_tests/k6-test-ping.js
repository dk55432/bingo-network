import ws from 'k6/ws';
import { check, sleep } from 'k6';

export const options = {
  vus: 20,
  duration: '30s',
};

const WS_URL = 'ws://192.168.1.154:8000/ws';
const messagesPerClient = 3;

function normalizeToText(data) {
  if (data == null) return '';
  if (typeof data === 'string') return data;
  if (typeof data?.data === 'string') return data.data;
  try { return String(data); } catch { return ''; }
}

export default function () {
  ws.connect(WS_URL, {}, (socket) => {
    let sawPong = 0;

    socket.on('message', (data) => {
      const text = normalizeToText(data);
      // If pongs are JSON: {"type":"pong"}
      if (text.includes('"type":"pong"') || text.includes('pong')) {
        sawPong++;
        console.log('CLIENT got message:', text);
      }
    });

    for (let i = 0; i < messagesPerClient; i++) {
      const before = sawPong;

      socket.send(JSON.stringify({ type: 'ping' }));

      const timeoutMs = 3000;
      const start = Date.now();

      // Wait for handler to fire (sleep yields to the runtime)
      while (sawPong === before && (Date.now() - start) < timeoutMs) {
        sleep(0.05);
      }

      const gotThisPong = sawPong > before;

      check(gotThisPong, {
        'received pong for ping': () => gotThisPong,
      });

      // Extra short delay so the last message isn't racing close()
      sleep(0.05);
    }

    socket.close();
  });
}
