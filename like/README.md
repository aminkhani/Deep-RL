<!DOCTYPE html>
<html>
<head>
<style>
$prime: #ff6e48
$ciBlue: #00FFF1
$ciRed: #ff00aa
$ciGreen: #46fcb4
$second: #0c1016

body,
html
    align-items: center
    background-color: $second
    display: flex
    font-family: 'Roboto'
    font-size: 10px
    height: 100%
    justify-content: center
    margin: 0
    padding: 0
    width: 100%

*
    box-sizing: border-box

button
    border: none
    border-radius: 2rem
    display: flex
    justify-content: flex-start
    align-items: center
    flex-direction: row
    cursor: pointer
    padding: 0 2rem 0 2rem
    transition: 150ms all
    background-color: lighten($second, 10%)

    &.dislike
        .icon
            svg.prime
                animation: dislike 550ms forwards ease-in-out

    &.active
        background-color: $ciBlue
        .counter
            color: $second
        .icon
            svg
                color: $second
            svg.prime
                animation: like 550ms forwards ease-in-out

    .counter
        font-size: 3rem
        font-weight: 700
        color: white
        padding: 2rem 0 2rem 2rem
        transition: 150ms all

    .icon
        padding: 1rem
        transition: 150ms all
        position: relative
        width: 3rem
        height: 3rem

        svg
            color: white
            position: absolute
            top: 0
            left: 0
            width: 3rem
            height: 3rem

            &.normal
                z-index: 1
                opacity: 0.5
            &.prime
                z-index: 2

@keyframes dislike
    0%
        color: $second
        transform: translate(0, 0%)
    100%
        color: $ciBlue
        transform: translate(0, 300%) scale(0)

@keyframes like
    0%
        color: white
        transform: translate(0, 0%)
    100%
        color: $ciBlue
        transform: translate(0, -300%) scale(0)
</style>
</head>
<body>
<button id="like">
        <div class="icon">
            <svg class="prime" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
                <path fill="currentColor"
                    d="M462.3 62.6C407.5 15.9 326 24.3 275.7 76.2L256 96.5l-19.7-20.3C186.1 24.3 104.5 15.9 49.7 62.6c-62.8 53.6-66.1 149.8-9.9 207.9l193.5 199.8c12.5 12.9 32.8 12.9 45.3 0l193.5-199.8c56.3-58.1 53-154.3-9.8-207.9z">
                </path>
            </svg>
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
                <path fill="currentColor"
                    d="M462.3 62.6C407.5 15.9 326 24.3 275.7 76.2L256 96.5l-19.7-20.3C186.1 24.3 104.5 15.9 49.7 62.6c-62.8 53.6-66.1 149.8-9.9 207.9l193.5 199.8c12.5 12.9 32.8 12.9 45.3 0l193.5-199.8c56.3-58.1 53-154.3-9.8-207.9z">
                </path>
            </svg>
        </div>
        <div class="counter" id="couter"></div>
    </button>
  <script>
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
</script>
</body>
</html>
