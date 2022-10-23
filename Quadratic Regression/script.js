let x_data = [];
let y_data = [];
let a,b,c;

const learningRate = 0.3;
const optimizer = tf.train.sgd(learningRate);

function setup(){
    createCanvas(400,400);
    a = tf.variable(tf.scalar(random(1)));
    b = tf.variable(tf.scalar(random(1)));
    c = tf.variable(tf.scalar(random(1)));

}

function mousePressed(){
    let x = map(mouseX, 0, width, 0, 1);
    let y = map(mouseY, 0, height, 1, 0);
    x_data.push(x);
    y_data.push(y);
}

function f(x){

    const x_t = tf.tensor1d(x);
    let termA = x_t.square().mul(a);
    let termB = x_t.mul(b);
    let termC = c;

    return termA.add(termB).add(termC);
}

function loss(pred, labels){
    return (pred.sub(labels).square()).mean();
}


function regressionLine(){
    let xs = [];
    let ys = [];
    let px = [];
    let py = [];
    for(let i = 0; i < 400; i++){
        xs.push(i/400);
    }
    y_ts = f(xs);
    ys = y_ts.dataSync();


    if(xs.length>0){

        for(let i = 0; i < xs.length; i++){
            px[i] = map(xs[i], 0, 1, 0, width);
            py[i] = map(ys[i], 1, 0, 0, height);
        }
        for(let i = 0; i < xs.length; i++){
            point(px[i], py[i]);
        }
    }
    //console.log(px,py);

}

function draw(){
    background(0);
    stroke(255);
    strokeWeight(4);


    tf.tidy(() => {
        if(x_data.length > 0){
            const y_t = tf.tensor1d(y_data);
            optimizer.minimize( () => {
                return loss(f(x_data), y_t);
            })
    
        }
        for(let i = 0; i < x_data.length; i++){
            let px = map(x_data[i], 0, 1, 0, width);
            let py = map(y_data[i], 1, 0, 0, height);
            point(px, py);
        }
    })
    
    regressionLine();
}

