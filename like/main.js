
const button = document.querySelector('#like');
const counter = '#couter';
let state = false;
let like = 12;

writeCounter(like, counter);
setState(state, button);

function writeCounter(number, id) {
    const elm = button.querySelector(id);
    elm.innerHTML = number;
}

function setState(state, elm) {
    if (state) {
        elm.classList.remove('dislike');
        elm.classList.add('active');
    } else {
        elm.classList.remove('active');
        elm.classList.add('dislike');
    }
}

function addLike() {
    if (!state) {
        like++;
        state = true;
        writeCounter(like, counter);
        setState(state, button);
    } else {
        like--;
        state = false;
        writeCounter(like, counter);
        setState(state, button);
    }
}

button.addEventListener('click', addLike);



































