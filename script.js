const MAX_LEN = 3;
const MAX_WORDS = 10;
const HIDDEN_SIZE = 32;
const LR = 0.01;
const EPOCHS = 200;

const TRAIN_DATA = [
    ["sp", "ak", "wi"],
    ["gs","sp","bi","ak","du"],
    ["sk","en","ne"],
    ["gs","du","ne","na"],
    ["ec","sk","wi"],
    ["ec","bi","na"],
    ["ec","bi","sp","du","sk","na","ak","ne","en","wi"],
    ["en","ec","wi"]
];

function asciiVector(word, maxLen = MAX_LEN) {
    word = word.toLowerCase().slice(0, maxLen);
    const vec = Array.from(word).map(c => c.charCodeAt(0)/255);
    while(vec.length < maxLen) vec.push(0);
    return vec;
}

function wordsToMatrix(words) {
    const padded = words.slice(0, MAX_WORDS);
    while(padded.length < MAX_WORDS) padded.push("");
    return padded.map(w => asciiVector(w));
}

function sortedIndices(words) {
    return [...words.keys()].sort((a,b)=>words[a].toLowerCase().localeCompare(words[b].toLowerCase()));
}

function randMatrix(rows, cols) {
    const mat = [];
    for(let i=0;i<rows;i++){
        mat.push(Array.from({length: cols},()=>Math.random()-0.5));
    }
    return mat;
}

function relu(v) { return v.map(x=>Math.max(0,x)); }
function softmax(v) {
    const m = Math.max(...v);
    const exps = v.map(x=>Math.exp(x-m));
    const s = exps.reduce((a,b)=>a+b,0);
    return exps.map(x=>x/s);
}

let W1 = randMatrix(MAX_LEN,HIDDEN_SIZE);
let W2 = randMatrix(HIDDEN_SIZE,MAX_WORDS);

for(let epoch=0; epoch<EPOCHS; epoch++){
    let totalLoss = 0;
    for(const sample of TRAIN_DATA){
        const X = wordsToMatrix(sample);
        const y = sortedIndices(sample);

        for(let i=0;i<X.length;i++){
            const xVec = X[i];

            let h = Array(HIDDEN_SIZE).fill(0);
            for(let j=0;j<HIDDEN_SIZE;j++){
                for(let k=0;k<MAX_LEN;k++) h[j] += xVec[k]*W1[k][j];
                h[j] = Math.max(0,h[j]);
            }

            let out = Array(MAX_WORDS).fill(0);
            for(let j=0;j<MAX_WORDS;j++){
                for(let k=0;k<HIDDEN_SIZE;k++) out[j] += h[k]*W2[k][j];
            }

            const probs = softmax(out);

            if(i<y.length){
                const target = Array(MAX_WORDS).fill(0);
                target[y[i]] = 1;
                const grad = probs.map((p,j)=>2*(p-target[j]));
                for(let j=0;j<HIDDEN_SIZE;j++){
                    for(let k=0;k<MAX_WORDS;k++){
                        W2[j][k] -= LR*grad[k];
                    }
                }
                totalLoss += grad.reduce((a,b)=>a+b*b,0);
            }
        }
    }
    if((epoch+1)%50===0) console.log(`Epoch ${epoch+1}, Loss: ${totalLoss.toFixed(4)}`);
}

const input = document.getElementById("inputWords");
function wordScore(wordVec){
    let h = Array(HIDDEN_SIZE).fill(0);
    for(let j=0;j<HIDDEN_SIZE;j++){
        for(let k=0;k<MAX_LEN;k++) h[j] += wordVec[k]*W1[k][j];
        h[j] = Math.max(0,h[j]);
    }
    let out = Array(MAX_WORDS).fill(0);
    for(let j=0;j<MAX_WORDS;j++){
        for(let k=0;k<HIDDEN_SIZE;k++) out[j] += h[k]*W2[k][j];
    }
    return out.reduce((a,b)=>a+b,0);
}

function sortWords(){
    const inputValue = input.value.trim();
    const output = document.getElementById("output");
    if(!inputValue){
        output.innerHTML = "No input subjects";
        return;
    }

    const words = inputValue.split(/\s+/);
    const vecs = wordsToMatrix(words);
    const scores = vecs.map(v=>wordScore(v));
    const sortedIdx = [...words.keys()].sort((a,b)=>scores[a]-scores[b]);

    output.innerHTML = `
        <b>Input:</b> ${words.join(' ')}<br>
        <b>Sorted:</b> ${sortedIdx.map(i=>words[i]).join(' ')}
    `;
}

input.addEventListener("keydown", function(event){
    if(event.key === "Enter") sortWords();
});
