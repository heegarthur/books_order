

// Simpele 'neuraal netwerk'-achtige functie
// (gebaseerd op lettercodes)
function asciiVector(word, maxLen = 3) {
    word = word.toLowerCase().slice(0, maxLen);
    const vec = Array.from(word).map(c => c.charCodeAt(0) / 255);
    while (vec.length < maxLen) vec.push(0);
    return vec;
}

function wordScore(word) {
    // maakt een soort “score” van letters
    const vec = asciiVector(word);
    // weegt elke letter een beetje anders
    return vec[0] * 1.2 + vec[1] * 0.8 + vec[2] * 0.5;
}

function sortWords() {
    const input = document.getElementById('inputWords').value.trim();
    const output = document.getElementById('output');
    if (!input) {
        output.innerHTML = "Input subjects!";
        return;
    }

    const words = input.split(/\s+/);
    const sorted = [...words].sort((a, b) => wordScore(a) - wordScore(b));

    output.innerHTML = `
        <b>Input:</b> ${words.join(' ')}<br>
        <b>Sorted:</b> ${sorted.join(' ')}
      `;
}

const input = document.getElementById("inputWords");
input.addEventListener("keydown", function(event) {
    if(event.key === "Enter"){
        sortWords();
    }
});