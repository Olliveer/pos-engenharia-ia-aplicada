importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest");

const MODEL_PATH = `yolov5n_web_model/model.json`;
const LABELS_PATH = `yolov5n_web_model/labels.json`;
const INPUT_DIM = 640;
const CLASS_THRESHOLD = 0.4;

let _labels = [];
let _model = null;

async function loadModelAndLabels(params) {
  await tf.ready();

  _labels = await (await fetch(LABELS_PATH)).json();
  _model = await tf.loadGraphModel(MODEL_PATH);

  const dummyInput = tf.ones(_model.inputs[0].shape);
  await _model.executeAsync(dummyInput);
  tf.dispose(dummyInput);

  postMessage({
    type: "ready",
  });
}

/**
 * Pré-processa a imagem para o formato aceito pelo YOLO:
 * - tf.browser.fromPixels(): converte ImageBitmap/ImageData para tensor [H, W, 3]
 * - tf.image.resizeBilinear(): redimensiona para [INPUT_DIM, INPUT_DIM]
 * - .div(255): normaliza os valores para [0, 1]
 * - .expandDims(0): adiciona dimensão batch [1, H, W, 3]
 *
 * Uso de tf.tidy():
 * - Garante que tensores temporários serão descartados automaticamente,
 *   evitando vazamento de memória.
 */
function preProccessImage(input) {
  return tf.tidy(() => {
    const image = tf.browser.fromPixels(input);

    return tf.image
      .resizeBilinear(image, [INPUT_DIM, INPUT_DIM])
      .div(255)
      .expandDims(0);
  });
}

async function runInference(tensor) {
  const output = await _model.executeAsync(tensor);
  tf.dispose(tensor);

  // O formato de saída depende do modelo específico, mas geralmente inclui:
  // - boxes: coordenadas das caixas delimitadoras [num_boxes, 4]
  // - scores: confiança de cada caixa [num_boxes]
  // - classes: classe prevista para cada caixa [num_boxes]
  const [boxes, scores, classes] = output.slice(0, 3);
  const [boxesData, scoresData, classesData] = await Promise.all([
    boxes.data(),
    scores.data(),
    classes.data(),
  ]);

  output.forEach((t) => tf.dispose(t));

  return {
    boxes: boxesData,
    scores: scoresData,
    classes: classesData,
  };
}

function* proccessPredictions({ boxes, scores, classes }, width, height) {
  for (let i = 0; i < scores.length; i++) {
    // Filtra previsões com baixa confiança
    if (scores[i] < CLASS_THRESHOLD) continue;

    const label = _labels[classes[i]];

    if (label !== "kite") {
      continue;
    }

    let [x1, y1, x2, y2] = boxes.slice(i * 4, (i + 1) * 4);
    x1 *= width;
    x2 *= width;
    y1 *= height;
    y2 *= height;

    const boxWidth = x2 - x1;
    const boxHight = y2 - y1;

    const centerX = x1 + boxHight / 2;
    const centerY = y1 + boxHight / 2;

    yield {
      x: centerX,
      y: centerY,
      score: (scores[i] * 100).toFixed(2),
    };
  }
}

loadModelAndLabels();

self.onmessage = async ({ data }) => {
  if (data.type !== "predict") return;
  if (!_model) {
    return;
  }

  const input = preProccessImage(data.image);

  const { width, height } = data.image;
  const inferenceResults = await runInference(input);

  for (const prediction of proccessPredictions(
    inferenceResults,
    width,
    height,
  )) {
    postMessage({
      type: "prediction",
      ...prediction,
    });
  }
};

console.log("🧠 YOLOv5n Web Worker initialized");
