import fs from "fs";
import path from "path";
import Importer from "./importer";

try {
  (async () => {
    // Get path from command line arguments
    const args = process.argv.slice(2);

    if (args.length === 0) {
      console.error("Error: Please provide a path as argument");
      console.log("Usage: ts-node script.ts <path>");
      process.exit(1);
    }

    const p = args[0]; //path.resolve(args[0]); // Use first argument as path

    // Verify path exists
    if (!fs.existsSync(p)) {
      console.error(`Error: Path does not exist: ${p}`);
      process.exit(1);
    }

    console.log(`Processing directory: ${p}`);

    const gfs = fs.readdirSync(p);
    const logs = {
      file: "",
      chats: 0,
      commits: 0,
      padinfo: 0,
      comments: 0,
      comment_replies: 0,
      chat_scrolling: 0,
      pad_scrolling: 0,
      pad_session: 0,
      chat_visibility: 0,
      tab_visibility: 0,
    };
    const modIds: Set<number> = new Set<number>([1, 2, 3]);

    for (const gf of gfs) {
      if (gf.indexOf("group_") === -1) {
        continue;
      }
      const gfp = path.join(p, gf);
      const importer = new Importer(gfp, modIds);
      await importer.getAuthors();
      await importer.getReadonlyPadIds();
      await importer.run();
      logs.chats = logs.chats + importer.logs.chats;
      logs.commits = logs.commits + importer.logs.commits;
      logs.padinfo = logs.padinfo + importer.logs.padinfo;
      logs.comments = logs.comments + importer.logs.comments;
      logs.comment_replies =
        logs.comment_replies + importer.logs.comment_replies;
      logs.chat_scrolling = logs.chat_scrolling + importer.logs.chat_scrolling;
      logs.pad_scrolling = logs.pad_scrolling + importer.logs.pad_scrolling;
      logs.pad_session = logs.pad_session + importer.logs.pad_session;
      logs.chat_visibility =
        logs.chat_visibility + importer.logs.chat_visibility;
      logs.tab_visibility = logs.tab_visibility + importer.logs.tab_visibility;
    }
    console.log(logs);
    console.log("end");
  })();
} catch (error) {
  console.warn(error);
}
