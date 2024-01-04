const OpenAI = require("openai")

const openai = new OpenAI();

async function main() {
  const completion = await openai.chat.completions.create({
    messages: [{"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
    ],
    model: "gpt-3.5-turbo",
  });

  const assistantResponse = completion.choices[0].message.content;
  console.log(assistantResponse);
}

main();
