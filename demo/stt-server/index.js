import express from "express";
import speech from "@google-cloud/speech";
import bodyParser from "body-parser";
import logger from "morgan";
import cors from "cors";
import http from "http";
import { Server } from "socket.io";
import { textToCode } from "./textToCodeFunctions.js";

import { PhilipsHue } from "./blast.tds.js";
import { createThing } from "./blast.node.js";

globalThis.consumeThing = async function (mac) {
  const philipshue = await createThing(PhilipsHue, mac);

  return philipshue;
};

// Google Settings
const encoding = "LINEAR16";
const sampleRateHertz = 16000;

const request = {
  config: {
    encoding: encoding,
    sampleRateHertz: sampleRateHertz,
    languageCode: "en-US",
    enableWordTimeOffsets: true,
    enableAutomaticPunctuation: true,
    enableWordConfidence: true,
    enableSpeakerDiarization: true,
    model: "command_and_search",
    useEnhanced: true,
  },
  interimResults: true,
};

//Key file
process.env.GOOGLE_APPLICATION_CREDENTIALS = "./key.json";
const speechClient = new speech.SpeechClient();

let detectedSpeech;

// Server setup
const app = express();
app.use(cors());
app.use(logger("dev"));
app.use(bodyParser.json());
const server = http.createServer(app);

const io = new Server(server, {
  cors: {
    origin: "http://localhost:3000",
    methods: ["GET", "POST"],
  },
});

io.on("connection", (socket) => {
  globalThis.io = io;
  let recognizeStream = null;
  console.log("** a user connected - " + socket.id + " **\n");

  socket.on("execute", async (code) => {
    //let blastCode = await textToCode(detectedSpeech);
    
    let blastCode = "philipshue.writeProperty('brightness', 127);"
    console.log("GeneratedCode:", blastCode);
    let toEval;
    if (
      blastCode.split("(")[0] == "philipshue.writeProperty" ||
      blastCode.split("(")[0] == "philipshue.invokeAction"
    ) {
      toEval = `
        const mac = "CD23A5E6F3CD";
  
        const philipshue = await consumeThing(mac);
  
        await ${blastCode}
        process.exit(0); `;
    } else {
      toEval = `
        const mac = "CD23A5E6F3CD";
  
        const philipshue = await consumeThing(mac);
  
        let tmp = await ${blastCode}
        let status = await tmp.value();
        console.log(status);
  
        process.exit(0);  
        `;
    }

    io.emit("code", blastCode);
    io.emit("codeWithTemplate", toEval);

    // Execute Code
    
      const AsyncFunction = new Function(
        'return Object.getPrototypeOf(async function(){}).constructor'
      )();
      const func = new AsyncFunction(
        'f',
        `${toEval}
          f();`
      );
      await func();
     
    console.log("toEval:", toEval);
  });

  socket.on("disconnect", async () => {
    console.log("** user disconnected ** \n");
  });

  socket.on("startGoogleCloudStream", function (data) {
    startRecognitionStream(this, data);
  });

  socket.on("endGoogleCloudStream", function () {
    console.log("** ending google cloud stream **\n");
    stopRecognitionStream();
  });

  socket.on("send_audio_data", async (audioData) => {
    io.emit("receive_message", "Got audio data");
    if (recognizeStream !== null) {
      try {
        recognizeStream.write(audioData.audio);
      } catch (err) {
        console.log("Error calling google api " + err);
      }
    } else {
      console.log("RecognizeStream is null");
    }
  });

  function startRecognitionStream(client) {
    console.log("* StartRecognitionStream\n");
    try {
      recognizeStream = speechClient
        .streamingRecognize(request)
        .on("error", console.error)
        .on("data", (data) => {
          const result = data.results[0];
          const isFinal = result.isFinal;

          const transcription = data.results
            .map((result) => result.alternatives[0].transcript)
            .join("\n");
          detectedSpeech = transcription;
          client.emit("receive_audio_text", {
            text: transcription,
            isFinal: isFinal,
          });

          // if end of utterance, let's restart stream
          // this is a small hack. After 65 seconds of silence, the stream will still throw an error for speech length limit
          if (data.results[0] && data.results[0].isFinal) {
            stopRecognitionStream();
            startRecognitionStream(client);
            console.log("restarted stream server side");
          }
        });
    } catch (err) {
      console.error("Error streaming google api " + err);
    }
  }

  function stopRecognitionStream() {
    if (recognizeStream) {
      recognizeStream.end();
    }
    recognizeStream = null;

    console.log("* StopRecognitionStream \n");
  }
});

server.listen(8081, () => {
  console.log("WebSocket server listening on port 8081.");
});
