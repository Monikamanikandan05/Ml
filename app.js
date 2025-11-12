const MODEL_URL='';const CLASS_NAMES=['Stop','Yield','Speed limit 30','Turn right','Turn left','No entry'];
let model=null;async function loadModel(){if(!MODEL_URL)return;model=await tf.loadGraphModel(MODEL_URL);}
function preprocess(img){return tf.tidy(()=>tf.image.resizeBilinear(tf.browser.fromPixels(img),[128,128]).div(255).expandDims(0));}
async function predict(img){if(!model)return;const p=model.predict(preprocess(img));const data=await p.data();
const i=data.indexOf(Math.max(...data));document.getElementById('lastPred').textContent=CLASS_NAMES[i];
document.getElementById('lastConf').textContent=(data[i]*100).toFixed(1)+'%';}
async function startCamera(){const s=await navigator.mediaDevices.getUserMedia({video:true});const v=document.getElementById('webcam');
v.srcObject=s;v.play();setInterval(()=>predict(v),1000);}document.getElementById('startBtn').onclick=startCamera;