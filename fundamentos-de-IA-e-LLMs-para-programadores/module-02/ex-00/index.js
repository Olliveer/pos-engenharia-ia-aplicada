import tf from "@tensorflow/tfjs-node";

async function trainModel(inputXs, outputYs) {
  const model = tf.sequential();

  // primeira camada da rede:
  // first layer of the network:
  // inputShape: 7 (idade + 3 cores + 3 localizações)

  // 80 neuronios = 80 pq tem pouca base de treino, quanto mais neuronios,
  // mais complexidade a rede pode aprender e consequentemente mais processamento vai usar

  // a ReLu age como um filtro, ela só deixa passar os valores positivos, os negativos são convertidos para 0
  // se a entrada for negativa, a saída é 0, se for positiva, a saída é o próprio valor.
  // Isso ajuda a rede a aprender padrões não lineares e evita o problema de gradientes desaparecendo durante o treinamento.
  model.add(
    tf.layers.dense({ inputShape: [7], units: 80, activation: "relu" }),
  );

  // saida: 3 neuronios (premium, medium, basic)
  // activation softmax é usada para classificação,
  // ela converte os valores de saída em probabilidades, onde a soma de todas as saídas é igual a 1.
  // Isso é útil para problemas de classificação, onde queremos prever a probabilidade de cada classe.
  model.add(tf.layers.dense({ units: 3, activation: "softmax" }));

  // Compilamos o modelo, definindo o otimizador, a função de perda e as métricas de avaliação.
  // [adam] = Adaptive Moment Estimation,
  // é um otimizador que combina as vantagens de outros otimizadores como o AdaGrad e o RMSProp,
  // ajustando os pesos da rede de forma eficiente durante o treinamento.
  // [categoricalCrossentropy] é uma função de perda usada para problemas de classificação multiclasse,
  // ela mede a diferença entre as distribuições de probabilidade previstas pelo modelo e as distribuições reais dos dados de treinamento.
  model.compile({
    optimizer: "adam", // algoritmo de otimização para ajustar os pesos da rede
    loss: "categoricalCrossentropy", // função de perda para problemas de classificação multiclasse
    metrics: ["accuracy"], // métrica para avaliar o desempenho do modelo durante o treinamento
  });

  // Treinamos o modelo
  await model.fit(inputXs, outputYs, {
    epochs: 100, // número de vezes que o modelo vai passar por todo o conjunto de dados de treinamento
    shuffle: true, // embaralha os dados a cada época para melhorar a generalização do modelo
    // batchSize: 32, // número de amostras que serão processadas antes de atualizar os pesos do modelo
    verbose: 0, // nível de verbosidade durante o treinamento (0 = silencioso, 1 = barra de progresso, 2 = uma linha por época)
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(
          `Epoch ${epoch}: loss = ${logs.loss.toFixed(4)}, accuracy = ${logs.acc.toFixed(4)}`,
        );
      },
    },
  });

  return model;
}

async function predict(model, pessoa) {
  // Convertendo a entrada para um tensor
  const tfInput = tf.tensor2d(pessoa);

  // Fazendo a previsão
  const prediction = model.predict(tfInput);

  const predArray = await prediction.array();

  return predArray[0].map((prob, index) => ({
    prob,
    index,
  }));
}

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
  [0.33, 1, 0, 0, 1, 0, 0], // Erick
  [0, 0, 1, 0, 0, 1, 0], // Ana
  [1, 0, 0, 1, 0, 0, 1], // Carlos
];

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
  [1, 0, 0], // premium - Erick
  [0, 1, 0], // medium - Ana
  [0, 0, 1], // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado);
const outputYs = tf.tensor2d(tensorLabels);

// inputXs.print();
// outputYs.print();

// quanto mais dado tivermos, melhor o modelo vai aprender, mas também vai levar mais tempo para treinar
const models = await trainModel(inputXs, outputYs);

const pessoa = {
  nome: "José",
  idade: 33,
  cor: "preto",
  localizacao: "Curitiba",
};

// Normalizamos os dados de entrada para a nova pessoa
// idade normalizada: (idade - idade_min) / (idade_max - idade_min)

const pessoaNormalizada = [
  [
    0.2, // idade normalizada (33 - 25) / (40 - 25)
    0, // azul
    0, // vermelho
    0, // verde
    0, // São Paulo
    0, // Rio
    1, // Curitiba
  ],
];

const predictions = await predict(models, pessoaNormalizada);
const results = predictions
  .sort((a, b) => b.prob - a.prob)
  .map(
    (pred) => `${labelsNomes[pred.index]}: ${(pred.prob * 100).toFixed(2)}%`,
  );

console.log("Predições ordenadas por probabilidade:");
console.log(results);
