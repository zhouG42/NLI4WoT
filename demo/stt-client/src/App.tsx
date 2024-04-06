import React from "react";
import AudioToText from "./AudioToText";
import Container from "react-bootstrap/Container";

function App() {
  return (
    <Container className="py-5 text-center">
      <h1>Natural Language Interface Demonstration</h1>
      <AudioToText />
    </Container>
  );
}

export default App;
