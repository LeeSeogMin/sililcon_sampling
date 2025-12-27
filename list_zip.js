const fs = require('fs');
const path = require('path');
const admZip = require('adm-zip'); // Let's check if adm-zip is available, or use jszip if I saw it.

// I saw jszip in the list_dir earlier.
const JSZip = require('jszip');

const zipPath = path.resolve('ms-word/src/Format.docx');

fs.readFile(zipPath, function(err, data) {
    if (err) throw err;
    JSZip.loadAsync(data).then(function(zip) {
        console.log('Zip Content:');
        Object.keys(zip.files).forEach(function(filename) {
            console.log(filename);
        });
    });
});
