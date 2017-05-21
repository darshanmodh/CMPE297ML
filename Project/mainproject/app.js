/**
 * Copyright 2015 IBM Corp. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

'use strict';
  var PythonShell = require('python-shell');
var express = require('express'),
  app = express(),
  util = require('util'),
  extend = util._extend,
  watson = require('watson-developer-cloud'),
  Q = require('q'),
  isString = function(x) {
    return typeof x === 'string';
  };

var MongoClient = require('mongodb').MongoClient;
var MongoDb = null;
// Connect to the db
MongoClient.connect("mongodb://cmpe272:cmpe272@ds021182.mlab.com:21182/cmpe272", function(err, db) {
  console.log('Mongo DB connected');
  if(err) { return console.dir(err); }
MongoDb = db;
});


// Bootstrap application settings
require('./config/express')(app);



app.get('/test', function(req,  res) {

    res.send("testCtrl Called");



});

app.get('/', function(req, res) {

  res.render('index');

});

app.get('/stock', function(req, res) {

  var symbol = req.query.symbol;
 
  var options = {
    mode: 'text',
    args: [symbol]
    
  };
   
  PythonShell.run('demo_new.py', options, function (err, results) {
    if (err) console.log(err);
    // results is an array consisting of messages collected during execution 
    console.log(results);
    console.log('results: %j', results);
    res.render('impact/stock', {title: "Sdf"});
  });

  

});
app.get('/corn', function(req, res) {

  MongoDb.collection('cropData', function(err, collection) {
    var options = {
      "sort": "year"
    };
    collection.find({"crop":"corn"},options, function(err, cursor) {
      cursor.toArray(function(err, items) {
        console.log(items);
        res.render('impact/corn', {items: items});
      });
    });
    
  });
  

});

app.get('/grape', function(req, res) {

  MongoDb.collection('cropData', function(err, collection) {
    var options = {
      "sort": "year"
    };
    collection.find({"crop":"grape"},options, function(err, cursor) {
      cursor.toArray(function(err, items) {
        console.log(items);
        res.render('impact/grape', {items: items});
      });
    });
    
  });
  

});

app.get('/cotton', function(req, res) {

  MongoDb.collection('cropData', function(err, collection) {
    var options = {
      "sort": "year"
    };
    collection.find({"crop":"cotton", "region":"usa"},options, function(err, cursor) {
      cursor.toArray(function(err, items) {
        console.log(items);
        res.render('impact/cotton', {items: items});
      });
    });
    
  });
  

});

app.get('/winegrape', function(req, res) {

  MongoDb.collection('cropData', function(err, collection) {
    var options = {
      "sort": "year"
    };
    collection.find({"crop":"wine grape"},options, function(err, cursor) {
      cursor.toArray(function(err, items) {
        console.log(items);
        res.render('impact/winegrape', {items: items});
      });
    });
    
  });
  

});

app.get('/api/getcropdata', function(req, res) {

  var crop = req.query.crop;

  MongoDb.collection('cropData', function(err, collection) {
    var options = {
      "sort": "year"
    };
    collection.find({"crop":crop},options, function(err, cursor) {
      cursor.toArray(function(err, items) {
        console.log(items);
        res.json({items: items});
      });
    });
    
  });
  

});


// error-handler settings
require('./config/error-handler')(app);

var port = process.env.VCAP_APP_PORT || 3001;
app.listen(port);
console.log('listening at:', port);
