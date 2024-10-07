// reference: 
// https://stackoverflow.com/a/26628904

function saveCheckboxState(checkboxId) {
    var checkbox = document.getElementById(checkboxId);
    localStorage.setItem(checkboxId, checkbox.checked);
}

function loadCheckboxState(checkboxId) {
    var checkbox = document.getElementById(checkboxId);
    var isChecked = localStorage.getItem(checkboxId) === 'true';
    checkbox.checked = isChecked;
}

window.onload = function() {
    loadCheckboxState('age');
    loadCheckboxState('ops');

    // set checkboxes according to parameters
    // reference:
    // https://stackoverflow.com/a/8206578

    var urlParams = new URLSearchParams(window.location.search);

    if (urlParams.has('age')) {
        document.getElementById('age').checked = urlParams.get('age') === 'true';
    } else {
        document.getElementById('age').checked = urlParams.get('age') === 'false';
    }

    if (urlParams.has('ops')) {
        document.getElementById('ops').checked = urlParams.get('ops') === 'true';
    } else {
        document.getElementById('ops').checked = urlParams.get('ops') === 'false';
    }
    
};

// toggle parameters

function generateURL() {
    // var urlRaw = window.location.href;
    // var url = urlRaw.split('?')[0] + "?" // reference: https://bobbyhadz.com/blog/javascript-remove-querystring-from-url
    var url = '/' + document.getElementById("caseID").innerText + '?';

    var age = document.getElementById("age");
    var ops = document.getElementById("ops");

    // make sure at least one is checked
    if (!age.checked && !ops.checked) {
        age.checked = true;
    }

    // update URL
    if (age.checked) {
        url += "age=" + age.value + "&";
    }
    if (ops.checked) {
        url += "ops=" + ops.value + "&";
    }
    if (url.slice(-1) === '&') {
        url = url.slice(0, -1);
    }

    saveCheckboxState('age');
    saveCheckboxState('ops');

    window.location.href = url;
}