let x_data = [];
let y_data = [];
let m,b;

const learningRate = 0.2;
const optimizer = tf.train.sgd(learningRate);

function setup(){
    createCanvas(400,400);
    m = tf.variable(tf.scalar(random(1)));
    b = tf.variable(tf.scalar(random(1)));

}

function mousePressed(){
    let x = map(mouseX, 0, width, 0, 1);
    let y = map(mouseY, 0, height, 1, 0);
    x_data.push(x);
    y_data.push(y);
}

function f(x){

    const x_t = tf.tensor1d(x);

    return (x_t.mul(m)).add(b);
}

function loss(pred, labels){
    return (pred.sub(labels).square()).mean();
}


function regressionLine(){
    let xs = [0,1]
    let y_ts = f(xs);
    let ys = y_ts.dataSync();
    let px = [map(xs[0], 0, 1, 0, width), map(xs[1], 0, 1, 0, width)];
    let py = [map(ys[0], 1, 0, 0, height), map(ys[1], 1, 0, 0, height)];
    line(px[0], py[0], px[1], py[1]);
    y_ts.dispose();
    

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

