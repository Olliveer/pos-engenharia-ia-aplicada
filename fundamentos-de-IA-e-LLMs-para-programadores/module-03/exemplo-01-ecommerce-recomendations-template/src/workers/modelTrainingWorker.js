import "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js";
import { workerEvents } from "../events/constants.js";

console.log("Model training worker initialized");
let _globalCtx = {};

const WEIGHTS = {
  category: 0.4,
  color: 0.3,
  price: 0.2,
  age: 0.1,
};

const normalize = (valeu, min, max) => (valeu - min) / (max - min) || 1;

function makeContext(products, users) {
  const ages = users.map((user) => user.age);
  const prices = products.map((p) => p.price);

  const minAge = Math.min(...ages);
  const maxAge = Math.max(...ages);

  const minPrice = Math.min(...prices);
  const maxPrice = Math.max(...prices);

  const colors = [...new Set(products.map((p) => p.color))];

  const categories = [...new Set(products.map((p) => p.category))];

  const colorsIndex = Object.fromEntries(colors.map((c, i) => [c, i]));
  const categoriesIndex = Object.fromEntries(categories.map((c, i) => [c, i]));

  // computar a média de idade dos usuários por produto (ajuda a personalizar recomendações)
  const midAge = (minAge + maxAge) / 2;
  const ageSums = {};
  const ageCounts = {};
  users.forEach((user) => {
    user.purchases.forEach((product) => {
      ageSums[product.name] = (ageSums[product.name] || 0) + user.age;
      ageCounts[product.name] = (ageCounts[product.name] || 0) + 1;
    });
  });

  const productAvgAgeNorm = Object.fromEntries(
    products.map((product) => {
      const avg = ageCounts[product.name]
        ? ageSums[product.name] / ageCounts[product.name]
        : midAge;

      return [product.name, normalize(avg, minAge, maxAge)];
    }),
  );

  return {
    products,
    users,
    colorsIndex,
    categoriesIndex,
    productAvgAgeNorm,
    minAge,
    maxAge,
    minPrice,
    maxPrice,
    numCategories: categories.length,
    numColors: colors.length,
    // price and age are normalized to [0, 1], and we have one-hot encoding for colors and categories, so the total input dimension is:
    // price + age + one-hot colors + one-hot categories
    dimentions: 2 + colors.length + categories.length,
  };
}

const oneHotWeighted = (index, length, weight) =>
  tf.oneHot(index, length).cast("float32").mul(weight);

function encodeProduct(product, ctx) {
  //normalizando dados para ficar entre 0 e 1, e aplicando pesos para cada feature
  const price = tf.tensor1d([
    normalize(product.price, ctx.minPrice, ctx.maxPrice) * WEIGHTS.price,
  ]);

  const age = tf.tensor1d([
    (ctx.productAvgAgeNorm[product.name] ?? 0.5) * WEIGHTS.age,
  ]);

  const category = oneHotWeighted(
    ctx.categoriesIndex[product.category],
    ctx.numCategories,
    WEIGHTS.category,
  );

  const color = oneHotWeighted(
    ctx.colorsIndex[product.color],
    ctx.numColors,
    WEIGHTS.color,
  );

  return tf.concat1d([price, age, category, color]);
}

function encodeUser(user, ctx) {
  if (user.purchases.length) {
    return tf
      .stack(user.purchases.map((product) => encodeProduct(product, ctx)))
      .min(0)
      .reshape([1, ctx.dimentions]);
  }
}

function createTrainingData(ctx) {
  const inputs = [];
  const labels = [];
  ctx.users.forEach((user) => {
    const userVector = encodeUser(user, ctx).dataSync();
    ctx.products.forEach((product) => {
      const productVector = encodeProduct(product, ctx).dataSync();

      const label = user.purchases.some((p) => p.name === product.name) ? 1 : 0;

      // Aqui você pode criar um dataset de treinamento usando userVector, productVector e label
      // Por exemplo, você pode concatenar userVector e productVector para criar uma entrada combinada
      // E usar label como a saída esperada para treinar um modelo de classificação binária
      // Exemplo de como criar uma entrada combinada:
      inputs.push([...userVector, ...productVector]);
      labels.push(label);
    });
  });

  return {
    xs: tf.tensor2d(inputs),
    ys: tf.tensor2d(labels, [labels.length, 1]),
    // A dimensão de entrada para o modelo seria a soma das dimensões do vetor do usuário e do vetor do produto
    inputDimention: ctx.dimentions * 2,
  };
}

async function trainModel({ users }) {
  console.log("Training model with users:", users);

  postMessage({
    type: workerEvents.progressUpdate,
    progress: { progress: 50 },
  });

  const productsData = await (await fetch("/data/products.json")).json();
  const context = makeContext(productsData, users);

  _globalCtx = context;

  context.productVectors = productsData.map((product) => {
    return {
      name: product.name,
      meta: { ...product },
      vector: encodeProduct(product, context).dataSync(),
    };
  });

  _globalCtx = context;

  const trainingData = createTrainingData(context);

  debugger;

  postMessage({
    type: workerEvents.trainingLog,
    epoch: 1,
    loss: 1,
    accuracy: 1,
  });

  setTimeout(() => {
    postMessage({
      type: workerEvents.progressUpdate,
      progress: { progress: 100 },
    });
    postMessage({ type: workerEvents.trainingComplete });
  }, 1000);
}
function recommend(user, ctx) {
  console.log("will recommend for user:", user);
  // postMessage({
  //     type: workerEvents.recommend,
  //     user,
  //     recommendations: []
  // });
}

const handlers = {
  [workerEvents.trainModel]: trainModel,
  [workerEvents.recommend]: (d) => recommend(d.user, _globalCtx),
};

self.onmessage = (e) => {
  const { action, ...data } = e.data;
  if (handlers[action]) handlers[action](data);
};
