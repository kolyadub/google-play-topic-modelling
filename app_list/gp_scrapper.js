var gplay = require('google-play-scraper');
var num = 50;
gplay.list({
    collection: gplay.collection.TOP_FREE,
    country: 'ru',
    num: num
})
    .then(response => {
        var ids = [];
        Object.keys(response).forEach(function (key) {
            ids.push(response[key]['appId']);
        }
        );

        fs = require('fs');
        fs.writeFile('app_list/top_free.txt', ids, function (err) {
            if (err) return console.log(err);
            console.log('top_free.txt created');
        });

    });

gplay.list({
    collection: gplay.collection.TOP_PAID,
    country: 'ru',
    num: num
})
    .then(response => {
        var ids = [];
        Object.keys(response).forEach(function (key) {
            ids.push(response[key]['appId']);
        }
        );

        fs = require('fs');
        fs.writeFile('app_list/top_paid.txt', ids, function (err) {
            if (err) return console.log(err);
            console.log('top_paid.txt created');
        });

    });
