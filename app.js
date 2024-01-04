const OpenAI = require("openai");
const dotenv = require("dotenv");
const fs = require("fs");

dotenv.config();
const openai = new OpenAI(
    {
        apiKey: process.env.OPENAI_API_KEY
    });

async function main(){
    try{
        const response = await openai.audio.transcriptions.create({
            model : "whisper-1",
            file : fs.createReadStream("./27 Facts That Will Make You Question Your Existence.mp3")
        })
        console.log(response);
    }
    catch(err){
        console.log(err);
    }
}
main();