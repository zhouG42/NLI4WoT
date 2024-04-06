/* eslint-disable react-hooks/exhaustive-deps */
import { default as React, useEffect, useState, useRef } from "react";
import { Button } from "react-bootstrap";
import Container from "react-bootstrap/Container";
import * as io from "socket.io-client";

const sampleRate = 16000;

const getMediaStream = () =>
  navigator.mediaDevices.getUserMedia({
    audio: {
      deviceId: "default",
      sampleRate: sampleRate,
      sampleSize: 16,
      channelCount: 1,
    },
    video: false,
  });

interface WordRecognized {
  final: boolean;
  text: string;
}

const AudioToText: React.FC = () => {
  const [connection, setConnection] = useState<io.Socket>();
  const [currentRecognition, setCurrentRecognition] = useState<string>();
  const [generatedCode, setGeneratedCode] = useState<string>();
  const [codeWithTemplate, setGeneratedCodeWithTemplate] = useState<string>();
  const [recognitionHistory, setRecognitionHistory] = useState<string[]>([]);
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [recorder, setRecorder] = useState<any>();
  const processorRef = useRef<any>();
  const audioContextRef = useRef<any>();
  const audioInputRef = useRef<any>();

  const speechRecognized = (data: WordRecognized) => {
    if (data.final) {
      setCurrentRecognition("");
      setRecognitionHistory((old) => [data.text, ...old]);
    } else setCurrentRecognition(data.text + "...");
  };

  const connect = () => {
    connection?.disconnect();
    setGeneratedCode("");
    setGeneratedCodeWithTemplate("");
    setCurrentRecognition("");
    const socket = io.connect("http://localhost:8081");
    socket.on("connect", () => {
      console.log("connected", socket.id);
      setConnection(socket);
    });

    socket.emit("startGoogleCloudStream");

    socket.on("receive_audio_text", (data) => {
      speechRecognized(data);
      console.log("received audio text", data);
    });

    socket.on("disconnect", () => {
      console.log("disconnected", socket.id);
    });

    socket.on("codeWithTemplate", (data) => {
      // Remove whitespace where 2 or more spaces are found
      data = data.replace(/\s{2,}/g, " ");
      // Add \n after each semicolon
      data = data.replace(/;/g, ";\n");
      // remove whitespace after \n 
      data = data.replace(/\n\s/g, "\n");
      
      
      setGeneratedCodeWithTemplate(data);
    });

    socket.on("code", (data) => {
      setGeneratedCode(data);
    });
  };

  const execute = () => {
    if (!connection) return;
    connection?.emit("execute");
    connection?.emit("endGoogleCloudStream");
    setConnection(undefined);
    setRecorder(undefined);
    setIsRecording(false);
  }

  const disconnect = () => {
    if (!connection) return;
    connection?.emit("endGoogleCloudStream");
    connection?.disconnect();
    processorRef.current?.disconnect();
    audioInputRef.current?.disconnect();
    audioContextRef.current?.close();
    setConnection(undefined);
    setRecorder(undefined);
    setIsRecording(false);
  };

  useEffect(() => {
    (async () => {
      if (connection) {
        if (isRecording) {
          return;
        }

        const stream = await getMediaStream();

        audioContextRef.current = new window.AudioContext();

        await audioContextRef.current.audioWorklet.addModule(
          "/src/worklets/recorderWorkletProcessor.js"
        );

        audioContextRef.current.resume();

        audioInputRef.current =
          audioContextRef.current.createMediaStreamSource(stream);

        processorRef.current = new AudioWorkletNode(
          audioContextRef.current,
          "recorder.worklet"
        );

        processorRef.current.connect(audioContextRef.current.destination);
        audioContextRef.current.resume();

        audioInputRef.current.connect(processorRef.current);

        processorRef.current.port.onmessage = (event: any) => {
          const audioData = event.data;
          connection.emit("send_audio_data", { audio: audioData });
        };
        setIsRecording(true);
      } else {
        console.error("No connection");
      }
    })();
    return () => {
      if (isRecording) {
        processorRef.current?.disconnect();
        audioInputRef.current?.disconnect();
        if (audioContextRef.current?.state !== "closed") {
          audioContextRef.current?.close();
        }
      }
    };
  }, [connection, isRecording, recorder]);

  return (
    <React.Fragment>
      <Container className="py-5 text-center">
        <Container fluid className="py-5 bg-primary text-light text-center ">
          <Container>
            <Button
              className={isRecording ? "btn-danger" : "btn-outline-light"}
              onClick={connect}
              disabled={isRecording}
            >
              Start Listening
            </Button>
            &nbsp;&nbsp;&nbsp;
            <Button
              className="btn-outline-light"
              onClick={execute}
              disabled={!isRecording}
            >
              Execute
            </Button>
            &nbsp;&nbsp;&nbsp;
            <Button
              className="btn-outline-light"
              onClick={disconnect}
              disabled={!isRecording}
            >
              Disconnect
            </Button>
          </Container>
        </Container>
        <Container className="py-5 text-center">
          {recognitionHistory.map((tx, idx) => (
            <p key={idx}>{tx}</p>
          ))}
          
          <p><b> Speech Input: </b>{currentRecognition}</p>
          <p><b>Generated Code: </b>{generatedCode}</p>
          <br></br>
          <br></br>
          <br></br>
          <br></br>
          <p>Code To Be Executed:</p>
          <textarea value={codeWithTemplate} readOnly></textarea>
        </Container>
      </Container>
    </React.Fragment>
  );
};

export default AudioToText;
