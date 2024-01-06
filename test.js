const youtubeUrl = "https://www.youtube.com/watch?v=XXYlFuWEuKI&list=RDQMgEzdN5RuCXE&start_radio=1";

// Create a URL object
const url = new URL(youtubeUrl);

// Get the video ID from the 'v' parameter
const videoId = url.searchParams.get("v");

console.log(videoId);
