import {inputWord2idx} from "./trained_model/mappings/input-word2idx.js";
import {wordContext} from "./trained_model/mappings/word-context.js";
import {targetWord2idx} from "./trained_model/mappings/target-word2idx.js";
import {targetIdx2word} from "./trained_model/mappings/target-idx2word.js";
import * as tf from "@tensorflow/tfjs-node";
import {stopwords} from "./stopWords.js";

const encoder = await tf.loadLayersModel(
  "file://./trained_model/tfjs/encoder/model.json"
);
const decoder = await tf.loadLayersModel(
  "file://./trained_model/tfjs/decoder/model.json"
);

// main function takes user input text and return parsed code
export async function textToCode(text) {
  const inputTensor = await convertSentenceToTensor(text);
  const states = encoder.predict(inputTensor);
  decoder.layers[1].resetStates(states);

  let responseTokens = [];
  let terminate = false;
  let nextTokenID = targetWord2idx["<SOS>"];
  let numPredicted = 0;

  while (!terminate) {
    const outputTokenTensor = tf.tidy(() => {
      const input = generateDecoderInputFromTokenID(nextTokenID);
      const prediction = decoder.predict(input);
      return prediction.squeeze().argMax();
    });

    const outputToken = await outputTokenTensor.data();
    outputTokenTensor.dispose();
    nextTokenID = Math.round(outputToken[0]);
    const word = targetIdx2word[nextTokenID];
    numPredicted++;

    if (word !== "<EOS>" && word !== "<SOS>") {
      responseTokens.push(word);
    }

    if (
      word === "<EOS>" ||
      numPredicted >= wordContext.decoder_max_seq_length
    ) {
      terminate = true;
    }

    await tf.nextFrame();
  }

  states[0].dispose();
  states[1].dispose();
  return responseTokens.join(" ");
}

// decoder current token output will be input for next time stamp
function generateDecoderInputFromTokenID(tokenID) {
  const buffer = tf.buffer([1, 1, wordContext.num_decoder_tokens]);
  buffer.set(1, 0, 0, tokenID);
  return buffer.toTensor();
}

function removeStopwords(str) {
  let res = [];
  let string = str.toLowerCase();
  let words = string.split(/[ .,]+/);
  for (let i = 0; i < words.length; i++) {
    let word_clean = words[i].split(".").join("");
    if (!stopwords.includes(word_clean)) {
      res.push(word_clean);
    }
  }
  return res.join(" ");
}

// convert input text into tensors
function convertSentenceToTensor(text) {
  let inputWordIds = [];
  let cleanedText = removeStopwords(text);
  console.log("cleaned text: ", cleanedText);
  let textArray = cleanedText.toString().split(" ");

  textArray.map((x) => {
    x = x.toLowerCase();
    let idx = "1"; // '1' index for UNK
    if (x in inputWord2idx) {
      idx = inputWord2idx[x];
    }
    inputWordIds.push(Number(idx));
  });

  if (inputWordIds.length > wordContext.encoder_max_seq_length){
    inputWordIds.length = wordContext.encoder_max_seq_length
    } else {
    let paddingArray = new Array((wordContext.encoder_max_seq_length - inputWordIds.length)).fill(0);
    console.log("paddingarray",paddingArray);
    inputWordIds = paddingArray.concat(inputWordIds)
}
  return tf.tensor2d(inputWordIds, [1, wordContext.encoder_max_seq_length]);
}
