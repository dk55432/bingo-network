import WebSocket from "ws";

const WS_URL = process.env.WS_URL || "ws://192.168.1.154:8000/ws";
const API_BASE = process.env.API_BASE || "http://192.168.1.154:8000";

// ---- configure your simulation ----
const GAME_ID = process.env.GAME_ID; // required
const NUM_PLAYERS = Number(process.env.NUM_PLAYERS || 20);
const CARDS_PER_PLAYER = Number(process.env.CARDS_PER_PLAYER || 3);

// Provide your transcribed card fixtures here.
// Format: array of cards; each card is 5x5 with integers or null for FREE.
var CARD_FIXTURES = JSON.parse(process.env.CARD_FIXTURES_JSON || "[]");
// Example card (put your real one in CARD_FIXTURES_JSON):
// [
//   [
//     [1,16,31,46,61],
//     [2,17,32,47,62],
//     [3,18,null,48,63],
//     [4,19,34,49,64],
//     [5,20,35,50,65]
//   ]
// ]

function pickCard(i) {
    // console.log("pickCards: CARD_FIXTURES=" +
    //     JSON.stringify(CARD_FIXTURES)
    // );
  if (CARD_FIXTURES.length === 0) {
    throw new Error("CARD_FIXTURES_JSON is empty. Provide at least one transcribed card.");
  }
  return CARD_FIXTURES[i % CARD_FIXTURES.length];
}

function makeGridsForPlayer(playerIndex) {
  // Create CARDS_PER_PLAYER cards for this player
  const cards = [];
  for (let c = 0; c < CARDS_PER_PLAYER; c++) {
    // Each player can reuse fixtures or vary them; this just rotates through your set
    cards.push(pickCard(playerIndex * 1000 + c));
  }
  return cards;
}

async function postCards({ gameId, playerId, grids }) {
  const res = await fetch(`${API_BASE}/cards`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ game_id: gameId, player_id: playerId, grids }),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`POST /cards failed: ${res.status} ${text}`);
  }
  return res.json(); // saved cards list
}

function wsConnect(url) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(url);
    ws.on("open", () => resolve(ws));
    ws.on("error", reject);
  });
}

function waitForMessage(ws, predicate, timeoutMs = 30000) {
  return new Promise((resolve, reject) => {
    const t = setTimeout(() => {
      cleanup();
      reject(new Error("Timed out waiting for WS message"));
    }, timeoutMs);

    function onMessage(data) {
      try {
        const text = data.toString();
        const msg = JSON.parse(text);
        if (predicate(msg)) {
          cleanup();
          resolve(msg);
        }
      } catch {
        // ignore non-JSON
      }
    }

    function onError(err) {
      cleanup();
      reject(err);
    }

    function cleanup() {
      clearTimeout(t);
      ws.off("message", onMessage);
      ws.off("error", onError);
    }

    ws.on("message", onMessage);
    ws.on("error", onError);
  });
}

async function joinPlayer({ displayName, gameId, playerIndex }) {
  const ws = await wsConnect(WS_URL);

  // You didn't give the server's exact join contract beyond:
  // {type:"join", display_name:"Joe", game_id:"some_UUID"} -> {type:"joined", player_id:"..."}
  ws.send(JSON.stringify({ type: "join", display_name: displayName, game_id: gameId }));

  const joined = await waitForMessage(
    ws,
    (m) => m && m.type === "joined" && typeof m.player_id === "string",
  );

  return { ws, playerId: joined.player_id, playerIndex };
}

function int_to_bingo_number(myint) {
    const letters = ['B', 'I', 'N', 'G', 'O'];
    if (myint < 1 || myint > 75) {
        throw new Error("Not a number 1..75: "+n);
    }
    const myletter = letters[ Math.floor((myint-1) / 15) ];
    return myletter + myint;
}

// Warning: does not guard against repeat numbers. (if you care about that)
function generate_random_bingo_cards(numcards) {
  let mycards = [];

  for (let c = 0; c < numcards; c++) {
    let mycard = [];

    for (let row = 0; row < 5; row++) {
      let myrow = [];

      // B: 1-15
      let cell = Math.floor(Math.random() * 15) + 1;
    //   console.log("Just created cell: ", cell);
      myrow.push(cell);

      // I: 16-30
      cell = Math.floor(Math.random() * 15) + 16;
      myrow.push(cell);

      // N: 31-45
      cell = Math.floor(Math.random() * 15) + 31;
      myrow.push(cell);

      // G: 46-60
      cell = Math.floor(Math.random() * 15) + 46;
      myrow.push(cell);

      // O: 61-75
      cell = Math.floor(Math.random() * 15) + 61;
      myrow.push(cell);

      mycard.push(myrow);
    }

    mycards.push(mycard);
  }

  return mycards;
}


async function simulate() {
  if (!GAME_ID) throw new Error("Set GAME_ID env var (UUID).");

  console.log(" ---- start host ----");
  const hostWs = await wsConnect(WS_URL);

  // If your host must send create/setup/start messages first, add them here.
  // For now we only listen for winners and then submit numbers.
   console.log("About to send host_reconnect");
   hostWs.send(JSON.stringify({ type: "host_reconnect", game_id: GAME_ID }));
    await waitForMessage(
        hostWs,
        (m) => m && m.type === "host_reconnected",
        30000,
    );

  let winnersReceived = false;
  const winnersPromise = waitForMessage(
    hostWs,
    (m) => m && m.type === "winner" && Array.isArray(m.winners),
    120000
  ).then((m) => {
    winnersReceived = true;
    return m;
  });

  console.log(" --- Host sets up a new game, receives the game_id")
   hostWs.send(JSON.stringify({ type: "setup_new_game" }));
    await waitForMessage(
        hostWs,
        (m) => m && m.type === "new_game",
        30000,
    );
    await waitForMessage(
        hostWs,
        (m) => m && m.type === "game_status",
        30000,
    );
    await waitForMessage(
        hostWs,
        (m) => m && m.type === "history",
        30000,
    );
  
  // -- create random cards
    CARD_FIXTURES = generate_random_bingo_cards(CARDS_PER_PLAYER);


  // ---- join players + register cards ----
  const players = [];
  for (let i = 0; i < NUM_PLAYERS; i++) {
    const displayName = `Player_${i + 1}`;
    players.push(joinPlayer({ displayName, gameId: GAME_ID, playerIndex: i }));
  }
  const joinedPlayers = await Promise.all(players);

  // Register cards via HTTP (transcribed numeric grids)
  // You can do this sequentially or concurrently; for load, we’ll do concurrent with a small limit.
  const concurrency = 10;
  let cursor = 0;

  console.log("--- About to register players cards.");
  async function worker() {
    while (cursor < joinedPlayers.length) {
      const myIndex = cursor++;
      const { playerId } = joinedPlayers[myIndex];

      const grids = makeGridsForPlayer(myIndex);
      await postCards({ gameId: GAME_ID, playerId, grids });
    }
  }

  const workers = Array.from({ length: Math.min(concurrency, joinedPlayers.length) }, worker);
  await Promise.all(workers);

  console.log(`Registered cards for ${joinedPlayers.length} players.`);

//   console.log(" --- Host sets the winning pattern to standard-bingo");
//    hostWs.send(JSON.stringify( {"type":"set_winning_pattern","pattern_name":"standard-bingo"} ));
  console.log(" --- Host sets the winning pattern to blackout");
   hostWs.send(JSON.stringify( {"type":"set_winning_pattern","pattern_name":"blackout"} ));
    await waitForMessage(
        hostWs,
        (m) => m && m.type === "game_status",
        // 30000,
        5 * 60 * 1000,
    );

   console.log(" --- Host starts the game")
   hostWs.send(JSON.stringify( {"type":"start_game"} ));
    await waitForMessage(
        hostWs,
        (m) => m && m.type === "game_status",
        30000,
    );

  // ---- host submits numbers until winners ----
  // Since your protocol doesn't include game_id in submit_number, the server presumably
  // attaches "current drawn state" to the active game already in SETUP/IN_PROGRESS.
  //
  // We'll just call 1..75 in order with a small delay.
  const submitDelayMs = 50;

  for (let n = 1; n <= 75; n++) {
//   for (let n = 1; n <= 5; n++) {
    if (winnersReceived) break;

    console.log("About to submit_number "
        + int_to_bingo_number(n)+" for number "+n
    );
    hostWs.send(JSON.stringify({ type: "submit_number", value: int_to_bingo_number(n) }));
    await new Promise((r) => setTimeout(r, submitDelayMs));
  }

  // Wait for winners broadcast (or timeout)
  const winnersMsg = await winnersPromise;
  console.log("Winners ("+winnersMsg.winners.length+"):", winnersMsg.winners);

  // Cleanup sockets
  hostWs.close();
  for (const p of joinedPlayers) p.ws.close();
}

simulate().catch((e) => {
  console.error("Simulation failed:", e);
  process.exit(1);
});
