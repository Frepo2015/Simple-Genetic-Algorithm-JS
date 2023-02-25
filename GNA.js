const f = require('fs');
const tf = require('@tensorflow/tfjs-node');

let fileP = "segP.txt";
//Funcion para Leer y dividir el archivo de pesos.
function leerFilePesosIniciales(file) {
    const contents = f.readFileSync(file, 'utf-8');
    const arr = contents.split(/\r?\n/);
    return arr;
}

//Importar datos de prueba
arregloPrueba = leerFilePesosIniciales(fileP);
numEjemplosP = parseInt(arregloPrueba[0]);
numAtributosP = parseInt(arregloPrueba[1]);
numClasesP = parseInt(arregloPrueba[2]);
console.log("Numero de ejemplos (Prueba): " + numEjemplosP + "\n" +
    "Numero de atributos (Prueba): " + numAtributosP + "\n" +
    "Numero de clases (Prueba): " + numClasesP);
let arregloXP = new Array(numEjemplosP);//Se declaran las matrices multidimenisionales
for (var i = 0; i < numEjemplosP; i++) {
    arregloXP[i] = arregloPrueba[i + 3].split(",").map(Number);
}
let arregloYP = new Array(numEjemplosP);
for (var i = 0; i < numEjemplosP; i++) {
    arregloYP[i] = arregloXP[i][numAtributosP];
    arregloXP[i].pop();
}
let XP = tf.tensor2d(arregloXP, [numEjemplosP, numAtributosP]);
let YP = tf.tensor2d(arregloYP, [numEjemplosP, 1]);
let YPc = tf.oneHot(tf.tensor1d(arregloYP, 'int32'), numClasesP);


//Inicializamos el numero de la Poblacion
const nPoblacion = 10;
let poblacion = [];

//Cargamos los pesos inciales de la red neuronal.
const filePesos = "pesosIniciales1.txt";
let pesos = leerFilePesosIniciales(filePesos); //Se devuelve el array con los pesos y los datos de la red
let capas = parseInt(pesos[0]); //Se almacena el numero de capas de al red
let lineas = capas * 2; //El numero de lineas del archivo que contienen los pesos
let arrCapas = [(capas + 1)]; // Se inicializa un array del tamaño del numero de capas, pata obtener el numero de neuronas por capa
let numElementos = 0;
let numElementosB=0;
for (var i = 0; i <= capas; i++) {//For para almacenar el # de Neuronas por capa
    arrCapas[i] = parseInt(pesos[i + 1]);
    //console.log(arrCapas[i]);
}
var arrAux = [capas];
let cromosoma1 = [];
let cromosoma1B = [];
//Se obtiene el primer cromosoma con los pesos iniciales
for (var i = 0; i < lineas; i++) {
    var aux = arrCapas[i] * arrCapas[i + 1];
    arrAux[i] = pesos[(i + (capas + 2))].split(",").map(Number);//Separa los strings por comas y los convierte en numeros
}
for (var i = 0; i < lineas; i++) {
    Array.prototype.push.apply(cromosoma1, arrAux[i]);
}
poblacion.push(cromosoma1);
//**************FUNCION FITNESS CROMOSOMA #1 ********************************
let XPCromo1 = tf.tensor2d(arrAux[0], [parseInt(pesos[1]), parseInt(pesos[2])]);//Tensor Capa1W
let XBCromo1 = tf.tensor1d(arrAux[1]); //Tensor Capa1 B
let XPCromo2 = tf.tensor2d(arrAux[2], [parseInt(pesos[2]), parseInt(pesos[3])]); //Tensor Capa
let XBCromo2 = tf.tensor1d(arrAux[3]);//Tensor Capa 2 B
let XPCromo3 = tf.tensor2d(arrAux[4], [parseInt(pesos[3]), parseInt(pesos[4])]);//Tensor Capa3W
let XBCromo3 = tf.tensor1d(arrAux[5]);//Tensor Capa3B
let fCapa = (XPCromo1.shape[0]*XPCromo1.shape[1]);
let fCapaB=(XBCromo1.shape[0]);
let sCapa = (XPCromo2.shape[0]*XPCromo2.shape[1]);
let sCapaB=(XBCromo2.shape[0]);
let tCapa = (XPCromo3.shape[0]*XPCromo3.shape[1]);
let tCapaB=(XBCromo3.shape[0]);

numElementos = fCapa+sCapa+tCapa+fCapaB+sCapaB+tCapaB;
console.log("numElementos: "+numElementos);

modelExtra = tf.sequential();
    hidden1 = tf.layers.dense({units: 4,inputShape: [numAtributosP],activation: 'relu'});
    modelExtra.add(hidden1);
    hidden2 = tf.layers.dense({units: 4,activation: 'sigmoid'});
    modelExtra.add(hidden2);
    output = tf.layers.dense({units: numClasesP,activation: 'softmax'});
    modelExtra.add(output);
    const learningRate = 0.05;
    const momentum = 0.005;
    const optimizador = tf.train.momentum(learningRate, momentum);
    modelExtra.compile({optimizer: optimizador,loss: 'meanSquaredError',metrics: ['accuracy'],});
    //console.log("Set Pesos Capa1");
    modelExtra.layers[0].setWeights([XPCromo1, XBCromo1]);
    //console.log("Set Pesos Capa2");
    modelExtra.layers[1].setWeights([XPCromo2,XBCromo2]);
    //console.log("Set Pesos Capa3");
    modelExtra.layers[2].setWeights([XPCromo3,XBCromo3]);
    arrPrecisiones = modelExtra.evaluate(XP,YPc);
    prueba1 = arrPrecisiones[0];
    prueba2 = arrPrecisiones[1];
    valorLoss = prueba1.dataSync();
    valorAcc = Math.round(((prueba2.dataSync()) * 100));
    console.log("\nLoss del individuo #1: " + valorLoss);
    console.log("Acc del individuo #1: " + valorAcc + "%\n");
    let precisiones = [];
    precisiones[0] =valorAcc; 
    //console.log("Conjunto de precisiones: "+precisiones)


var min = -1;
var max = 1;
//Se genera la poblacion de manera aleatoria y se obtiene la precision de cada individuo.
for (i = 1; i <= nPoblacion; i++) {
    let arrAux1 = [];
    let arrAux2 = [];
    for (j = 0; j < numElementos; j++) {
        arrAux1[j] = Math.random() * (max - min) + min;
    }
    poblacion.push(arrAux1);
//**************FUNCION FITNESS POBLACION INICIAL ********************************
let XPCromo1P = tf.tensor2d(poblacion[i].slice(0,fCapa), [parseInt(pesos[1]), parseInt(pesos[2])]);//Tensor Capa1W
let XBCromo1P = tf.tensor1d(poblacion[i].slice(fCapa,fCapa+fCapaB)); //Tensor Capa1 B
let XPCromo2P = tf.tensor2d(poblacion[i].slice(fCapa+fCapaB,fCapa+fCapaB+sCapa), [parseInt(pesos[2]), parseInt(pesos[3])]); //Tensor Capa
let XBCromo2P = tf.tensor1d(poblacion[i].slice(fCapa+fCapaB+sCapa,fCapa+fCapaB+sCapa+sCapaB));//Tensor Capa 2 B
let XPCromo3P = tf.tensor2d(poblacion[i].slice(fCapa+fCapaB+sCapa+sCapaB,fCapa+fCapaB+sCapa+sCapaB+tCapa), [parseInt(pesos[3]), parseInt(pesos[4])]);//Tensor Capa3W
let XBCromo3P = tf.tensor1d(poblacion[i].slice(fCapa+fCapaB+sCapa+sCapaB+tCapa,numElementos));//Tensor Capa3B
    modelExtra.layers[0].setWeights([XPCromo1P, XBCromo1P]);
    modelExtra.layers[1].setWeights([XPCromo2P,XBCromo2P]);
    modelExtra.layers[2].setWeights([XPCromo3P,XBCromo3P]);
    arrPrecisiones = modelExtra.evaluate(XP,YPc);
    prueba1 = arrPrecisiones[0];
    prueba2 = arrPrecisiones[1];
    valorLoss1 = prueba1.dataSync();
    valorAcc1 = Math.round(((prueba2.dataSync()) * 100));
    console.log("\nLoss del individuo #"+(i+1)+": " + valorLoss1);
    console.log("Acc del individuo "+(i+1)+": " + valorAcc1 + "%\n");
    precisiones.push(valorAcc1);
}
console.log(precisiones);

//************************************************  CRUZA  *******************************************
//console.log(poblacion[1]);
//console.log(poblacion[2]);
let generaciones = 0;
let con = 0;
let precicionesOrdenadas = [];
    for (let i = 0; i < precisiones.length; i++) { //Se 
        precicionesOrdenadas[i] = precisiones[i];
    }
do {//*************************REPETICION GENERACIONAL*****************************************
    precicionesOrdenadas.sort(function(a, b){return b - a});
    console.log("\n\n--------------------------------Generacion #"+(generaciones+1)+"------------------------------------------------")
    let parte1 = Math.round(poblacion[1].length*0.8); //********************Porcion de Cruza
    let parte2 = poblacion[1].length-parte1;
    //console.log(parte1);
    //console.log(parte2);
    let arrayHijo1=[];
    let arrayHijo2=[];
    //Seleccionamos los padres de manera aleatoria.
    for (let i = 0; i < precisiones.length; i++) {
        precicionesOrdenadas[i] = precisiones[i];
    }
    
    precicionesOrdenadas.sort(function(a, b){return b - a});
    //console.log(precicionesOrdenadas);
    let padre1=Math.round(Math.random()*(poblacion.length-1));
    let padre2=Math.round(Math.random()*(poblacion.length-1));
    //padre1=precisiones.indexOf(mayor);
    //padre2=precisiones.indexOf(sMayor);
    console.log("\nPadre 1 ("+padre1+")");
    console.log("Padre 2 ("+padre2+")\n");
//CRUZA
    Array.prototype.push.apply(arrayHijo1, poblacion[padre1].slice(0,parte1));
    Array.prototype.push.apply(arrayHijo1, poblacion[padre2].slice(parte1,numElementos));
    Array.prototype.push.apply(arrayHijo2, poblacion[padre2].slice(0,parte1));
    Array.prototype.push.apply(arrayHijo2, poblacion[padre1].slice(parte1,numElementos));
    //MUTACION
    porcentajeMutacion = Math.round(poblacion[0].length * 0.2);
        console.log("Mutacion: "+porcentajeMutacion);
        for (let m = 0; m < porcentajeMutacion; m++) {
            arrayHijo1[Math.round(Math.random() * ((numElementos-1) - 0) + 0)]=Math.random() * (max - min) + min;
            arrayHijo2[Math.round(Math.random() * ((numElementos-1) - 0) + 0)]=Math.random() * (max - min) + min;
        }
    //console.log(arrayHijo1.length);
    poblacion.push(arrayHijo1);
    //console.log(arrayHijo2.length);
    poblacion.push(arrayHijo2);
    //**************FUNCION FITNESS HIJO #1 ********************************
    let XPCromo1P = tf.tensor2d(arrayHijo1.slice(0,fCapa), [parseInt(pesos[1]), parseInt(pesos[2])]);//Tensor Capa1W
    let XBCromo1P = tf.tensor1d(arrayHijo1.slice(fCapa,fCapa+fCapaB)); //Tensor Capa1 B
    let XPCromo2P = tf.tensor2d(arrayHijo1.slice(fCapa+fCapaB,fCapa+fCapaB+sCapa), [parseInt(pesos[2]), parseInt(pesos[3])]); //Tensor Capa
    let XBCromo2P = tf.tensor1d(arrayHijo1.slice(fCapa+fCapaB+sCapa,fCapa+fCapaB+sCapa+sCapaB));//Tensor Capa 2 B
    let XPCromo3P = tf.tensor2d(arrayHijo1.slice(fCapa+fCapaB+sCapa+sCapaB,fCapa+fCapaB+sCapa+sCapaB+tCapa), [parseInt(pesos[3]), parseInt(pesos[4])]);//Tensor Capa3W
    let XBCromo3P = tf.tensor1d(arrayHijo1.slice(fCapa+fCapaB+sCapa+sCapaB+tCapa,numElementos));//Tensor Capa3B
        modelExtra.layers[0].setWeights([XPCromo1P, XBCromo1P]);
        modelExtra.layers[1].setWeights([XPCromo2P,XBCromo2P]);
        modelExtra.layers[2].setWeights([XPCromo3P,XBCromo3P]);
        arrPrecisiones = modelExtra.evaluate(XP,YPc);
        prueba1 = arrPrecisiones[0];
        prueba2 = arrPrecisiones[1];
        valorLoss2 = prueba1.dataSync();
        valorAcc2 = Math.round(((prueba2.dataSync()) * 100));
        console.log("\nLoss del hijo1 : " + valorLoss2);
        console.log("Acc del hijo1 : " + valorAcc2+ "%\n");
        precisiones.push(valorAcc2);
        
        
        //****************************FUNCION FITNESS HIJO #2*****************************
    XPCromo1P = tf.tensor2d(arrayHijo2.slice(0,fCapa), [parseInt(pesos[1]), parseInt(pesos[2])]);//Tensor Capa1W
    XBCromo1P = tf.tensor1d(arrayHijo2.slice(fCapa,fCapa+fCapaB)); //Tensor Capa1 B
    XPCromo2P = tf.tensor2d(arrayHijo2.slice(fCapa+fCapaB,fCapa+fCapaB+sCapa), [parseInt(pesos[2]), parseInt(pesos[3])]); //Tensor Capa
    XBCromo2P = tf.tensor1d(arrayHijo2.slice(fCapa+fCapaB+sCapa,fCapa+fCapaB+sCapa+sCapaB));//Tensor Capa 2 B
    XPCromo3P = tf.tensor2d(arrayHijo2.slice(fCapa+fCapaB+sCapa+sCapaB,fCapa+fCapaB+sCapa+sCapaB+tCapa), [parseInt(pesos[3]), parseInt(pesos[4])]);//Tensor Capa3W
    XBCromo3P = tf.tensor1d(arrayHijo2.slice(fCapa+fCapaB+sCapa+sCapaB+tCapa,numElementos));//Tensor Capa3B
        modelExtra.layers[0].setWeights([XPCromo1P, XBCromo1P]);
        modelExtra.layers[1].setWeights([XPCromo2P,XBCromo2P]);
        modelExtra.layers[2].setWeights([XPCromo3P,XBCromo3P]);
        arrPrecisiones = modelExtra.evaluate(XP,YPc);
        prueba1 = arrPrecisiones[0];
        prueba2 = arrPrecisiones[1];
        valorLoss3 = prueba1.dataSync();
        valorAcc3 = Math.round(((prueba2.dataSync()) * 100));
        console.log("\nLoss del hijo2 : " + valorLoss3);
        console.log("Acc del hijo2 : " + valorAcc3+ "%\n");
        precisiones.push(valorAcc3);
        //console.log(precisiones);
        //console.log(precicionesOrdenadas);
    //console.log(precicionesOrdenadas);
    if(precicionesOrdenadas[0]>=45){
        con = 1
    }
        generaciones = generaciones +1;
    //}while(generaciones!=10);
}while(con !=1);
console.log("("+precisiones.indexOf(precicionesOrdenadas[0])+")Precision mayor es: "+precicionesOrdenadas[0]);
console.log("El Cromosoma con la mejor solucion es: ["+poblacion[precisiones.indexOf(precicionesOrdenadas[0])]+"]");




