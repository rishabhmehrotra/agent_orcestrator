// Based on http://expressjs.com/en/starter/hello-world.html
const express = require('express');
const http = require('http');
const app = express();
const port = 4000;

app.use(express.json());

app.get('/', (req, res) => {
  const query = req.query.q;
  if (!query) {
    return res.status(400).send('Query parameter is required');
  }

  const postData = JSON.stringify({ query: query });

  const options = {
    hostname: '35.232.21.114',
    port: 8000,
    path: '/predict/multi_class_intent',
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Content-Length': postData.length
    }
  };

  const reqML = http.request(options, (resML) => {
    let data = '';

    resML.on('data', (chunk) => {
      data += chunk;
    });

    resML.on('end', () => {
      try {
        const result = JSON.parse(data);
        res.send(`Query: ${result.query}\nIntent: ${result.intent}\nScore: ${result.score}`);
      } catch (error) {
        console.error('Error parsing the ML model response:', error);
        res.status(500).send('Error parsing the ML model response');
      }
    });
  });

  reqML.on('error', (error) => {
    console.error('Error calling the ML model:', error);
    res.status(500).send('Error calling the ML model');
  });

  reqML.write(postData);
  reqML.end();
});

app.listen(port, () => {
  console.log(`Example app listening on port ${port}`);
});

