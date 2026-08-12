import fs from "node:fs";
import FormData from "form-data";

// node 18+ has global fetch; if not, install node-fetch
// npm i form-data

async function scanCard({ apiBase, imagePath, cornerPixels }) {
  const form = new FormData();
  form.append("file", fs.createReadStream(imagePath));
  form.append("corners", JSON.stringify(cornerPixels)); // tl/bl/tr/br

  const res = await fetch(`${apiBase}/scan-card`, {
    method: "POST",
    body: form,
    headers: form.getHeaders(), // important for multipart boundary
  });

  if (!res.ok) {
    throw new Error(`scan-card failed: ${res.status} ${await res.text()}`);
  }

  // Response shape from your endpoint:
  // return {"cards": cards}
  // where cards = [ ... ] (each card is a 5x5 grid with string labels like B12/I16/N32/G51/O63/FREE)
  const json = await res.json();
  return json; // { cards: [...] }
}



async function send_number(pendingNumber) {
    // broadcastButton click equivalent
    const message = {
    type: "submit_number",
    value: pendingNumber, // e.g. 17
    };
    socket.send(JSON.stringify(message));
}

async function postCards({ apiBase, gameId, playerId, grids }) {
  const res = await fetch(`${apiBase}/cards`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ game_id: gameId, player_id: playerId, grids }),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`POST /cards failed: ${res.status} ${text}`);
  }

  return await res.json(); // saved card(s) as returned by your endpoint
}

