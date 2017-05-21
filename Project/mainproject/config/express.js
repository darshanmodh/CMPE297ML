/**
 * Copyright 2015 IBM Corp. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

'use strict';

// Module dependencies
var express    = require('express'),
  bodyParser   = require('body-parser');
  var partials      = require('express-partials');

module.exports = function (app) {
  app.set('view engine', 'ejs');
  app.set('views', __dirname + '/../views/');//will work
  // Include partials middleware into the server
  app.use(partials());
  //require('ejs').delimiter = '$';
  app.enable('trust proxy');

  // Configure Express
  app.use(bodyParser.urlencoded({ extended: true, limit: '15mb' }));
  app.use(bodyParser.json({ limit: '15mb' }));
  app.use('/theme', express.static(__dirname + '/../theme/HTML'));
  app.use('/images', express.static(__dirname + '/../images'));
  app.use('/assets', express.static(__dirname + '/../assets'));

  // Only loaded when SECURE_EXPRESS is `true`
  if (process.env.SECURE_EXPRESS)
    require('./security')(app);

};
