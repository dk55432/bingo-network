// Run it

//     Single ping:

// bash

// node ws_check.mjs ws://10.0.0.153:8000/ws 1

//     10 pings:

// bash

// node ws_check.mjs ws://10.0.0.153:8000/ws 10 100

import WebSocket from "ws";

const url = process.argv[2] || "ws://10.0.0.153:8000/ws";
const count = Number(process.argv[3] ?? 1);     // number of pings
const intervalMs = Number(process.argv[4] ?? 200); // delay between pings

let receivedPongs = 0;
let closed = false;

const ws = new WebSocket(url);

const sendPing = (i) => {
  ws.send(JSON.stringify({ type: "ping" }));
  console.log(`sent ping ${i + 1}/${count}`);
};

ws.on("open", () => {
  if (count <= 1) {
    sendPing(0);
    return;
  }
  for (let i = 0; i < count; i++) {
    setTimeout(() => sendPing(i), i * intervalMs);
  }
});

ws.on("message", (data) => {
  const text = data.toString();
  // basic check; adjust if your server sends more message types
  if (text.includes('"type":"pong"') || text.trim() === '{"type":"pong"}') {
    receivedPongs++;
    console.log(`received pong (${receivedPongs}/${count})`);
  } else {
    console.log("received message:", text);
  }
});

ws.on("error", (e) => {
  console.error("ws error:", e);
  process.exit(2);
});

ws.on("close", () => {
  closed = true;
});

const timeoutMs = Math.max(2000, count * intervalMs + 1000);
setTimeout(() => {
  if (!closed) ws.close();

  const ok = receivedPongs >= count;
  console.log(ok ? "PASS" : "FAIL", { receivedPongs, expected: count });
  process.exit(ok ? 0 : 1);
}, timeoutMs);
