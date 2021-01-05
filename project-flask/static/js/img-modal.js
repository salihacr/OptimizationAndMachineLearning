function onClick(element) {
    document.getElementById("img01").src = element.src;
    document.getElementById("modal01").style.display = "block";
}
/*

const modal = document.getElementById('modal01');
const img = document.getElementById('img01');

modal.addEventListener("click", function () {
    img.className += " out";
    setTimeout(function () {
        modal.style.display = "none";
        img.className = "modal-content";
    }, 400);
});*/