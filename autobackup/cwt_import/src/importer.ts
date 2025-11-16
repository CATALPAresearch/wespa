import fs from "fs";
import DB from "./pg";
const CS = require("./lib/Changeset.js");

export default class Importer {
  public data: any;
  public authors: IAuthor = {};
  public readonlyIds: IReadonlyPad = {};
  public readonlyToPads: IReadonlyPad = {};
  private _modIds: Set<number>;
  public logs = {
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

  constructor(filePath: fs.PathOrFileDescriptor, modIds: Set<number>) {
    this.data = JSON.parse(fs.readFileSync(filePath).toString());
    this.logs.file = filePath.toString();
    this._modIds = modIds;
  }

  public async run(): Promise<void> {
    const todo = [];
    for (const entry of this.data) {
      const key = entry[0];
      const value = entry[1];
      if (key.indexOf("pad") !== -1 && key.indexOf("chat") !== -1) {
        todo.push(this.handleChat(key, value));
      } else if (key.indexOf("pad") !== -1 && key.indexOf("revs") !== -1) {
        todo.push(this.handleRev(key, value));
      } else if (key.indexOf("tracking") !== -1) {
        todo.push(this.handleTracking(key, value));
      } else if (key.indexOf("comment-replies") !== -1) {
        todo.push(this.handleCommentReplies(key, value));
      } else if (key.indexOf("comments") !== -1) {
        todo.push(this.handleComments(key, value));
      } else if (
        key.indexOf("pad") !== -1 &&
        key.indexOf("revs") === -1 &&
        key.indexOf("readonly2pad") === -1 &&
        key.indexOf("sessionstorage") === -1 &&
        key.indexOf("pad2readonly") === -1
      ) {
        todo.push(this.handlePadInfo(key, value));
      }
    }
    await Promise.all(todo);
  }

  public async handleRev(key: string, value: any) {
    try {
      const groupid = parseInt(key.split("_")[3]);
      const padid = key.split(":")[1];
      const rev = parseInt(key.split(":")[3]);
      const text = value.changeset;
      const authorid =
        value.meta &&
        typeof value.meta.author === "string" &&
        value.meta.author.indexOf("a.") !== -1
          ? value.meta.author
          : null;
      if (authorid === null) {
        return;
      }
      const timestamp = new Date(value.meta.timestamp);
      const taskid = parseInt(key.split("_")[1]);
      const userid = this.authors[authorid] ? this.authors[authorid] : null;
      const moderator = userid !== null ? this._modIds.has(userid) : false;
      const cs = CS.unpack(text);
      const ops = CS.opIterator(cs.ops);
      let inserts = 0;
      let deletes = 0;
      while (ops.hasNext()) {
        const next = ops.next();
        switch (next.opcode) {
          case "+":
            inserts += next.chars;
            break;
          case "-":
            deletes += next.chars;
            break;
        }
      }
      const d = [
        groupid,
        padid,
        rev,
        text,
        authorid,
        timestamp,
        taskid,
        userid,
        moderator,
        cs.oldLen,
        cs.newLen,
        inserts,
        deletes,
      ];
      await DB.query(
        "INSERT INTO pad_commit (groupid, padid, rev, text, authorid, timestamp, taskid, userid, moderator, textprevlen, textnewlen, textnumcharsadd, textnumcharsdel) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)",
        d
      );
      this.logs.commits++;
    } catch (error) {
      console.error("handleRev");
      console.error(error);
    }
  }

  public async handleChat(key: string, value: any) {
    try {
      const groupid = parseInt(key.split("_")[3]);
      const rev = parseInt(key.split(":")[3]);
      const text = value.text;
      const padid = key.split(":")[1];
      const authorid = value.userId;
      const userid = typeof this.authors[authorid]
        ? this.authors[authorid]
        : null;
      const timestamp = new Date(value.time);
      const taskid = parseInt(key.split("_")[1]);
      const moderator = userid !== null ? this._modIds.has(userid) : false;
      const d = [
        groupid,
        rev,
        text,
        padid,
        authorid,
        timestamp,
        taskid,
        userid,
        moderator,
      ];
      await DB.query(
        "INSERT INTO pad_chat (groupid, rev, text, padid, authorid, timestamp, taskid, userid, moderator) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
        d
      );
      this.logs.chats++;
    } catch (error) {
      console.error("handleChat");
      console.error(error);
    }
  }

  public async handleTracking(key: string, value: any) {
    try {
      let groupid: any = null;
      if (value.pad.indexOf("r.") !== -1) {
        const pad = this.readonlyToPads[value.pad];
        if (pad) {
          groupid = parseInt(pad.split("_")[3]);
        }
      } else {
        groupid = parseInt(value.pad.split("_")[1]);
      }
      const authorid =
        typeof value.user === "string" && value.user.indexOf("a.") !== -1
          ? value.user
          : null;
      const padid = value.pad;
      const tabid = value.tab;
      const sessop = parseInt(key.split(":")[3]);
      const timestamp = new Date(parseInt(key.split(":")[2]));
      let taskid: any = null;
      if (value.pad.indexOf("r.") !== -1) {
        const pad = this.readonlyToPads[value.pad];
        if (pad) {
          taskid = parseInt(pad.split("_")[1]);
        }
      } else {
        taskid = parseInt(value.pad.split("_")[1]);
      }
      const userid =
        authorid !== null && this.authors[authorid]
          ? this.authors[authorid]
          : null;
      const moderator = userid !== null ? this._modIds.has(userid) : false;
      if (value.type === 0) {
        const d = [
          groupid,
          authorid,
          padid,
          tabid,
          sessop,
          timestamp,
          "connect",
          userid,
          taskid,
          moderator,
        ];
        await DB.query(
          "INSERT INTO pad_session (groupid, authorid, padid, tabid, sessop, timestamp, state, userid, taskid, moderator) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)",
          d
        );
        this.logs.pad_session++;
      } else if (value.type === 2) {
        const state = value.state === true ? "visible" : "hidden";
        const d = [
          groupid,
          authorid,
          padid,
          tabid,
          sessop,
          timestamp,
          state,
          taskid,
          userid,
          moderator,
        ];
        await DB.query(
          "INSERT INTO pad_chat_visibility (groupid, authorid, padid, tabid, sessop, timestamp, state, taskid, userid, moderator) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)",
          d
        );
        this.logs.chat_visibility++;
      } else if (value.type === 3) {
        const state = value.state === true ? "visible" : "hidden";
        const d = [
          groupid,
          authorid,
          padid,
          tabid,
          sessop,
          timestamp,
          state,
          taskid,
          userid,
          moderator,
        ];
        await DB.query(
          "INSERT INTO pad_visibility (groupid, authorid, padid, tabid, sessop, timestamp, state, taskid, userid, moderator) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)",
          d
        );
        this.logs.tab_visibility++;
      } else if (value.type === 4) {
        const d = [
          groupid,
          authorid,
          padid,
          tabid,
          sessop,
          timestamp,
          "disconnect",
          userid,
          taskid,
          moderator,
        ];
        await DB.query(
          "INSERT INTO pad_session (groupid, authorid, padid, tabid, sessop, timestamp, state, userid, taskid, moderator) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)",
          d
        );
        this.logs.pad_session++;
      } else if (value.type === 5) {
        const top = value.state.top.index;
        const bottom = value.state.bottom.index;
        const count = value.state.count;
        const d = [
          groupid,
          authorid,
          padid,
          tabid,
          sessop,
          timestamp,
          top,
          bottom,
          taskid,
          userid,
          count,
          moderator,
        ];
        await DB.query(
          "INSERT INTO pad_scrolling (groupid, authorid, padid, tabid, sessop, timestamp, top, bottom, taskid, userid, count, moderator) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)",
          d
        );
        this.logs.pad_scrolling++;
      } else if (value.type === 7) {
        const top = value.state.up;
        const bottom = value.state.dwn;
        const mintop = value.state.min;
        const maxbottom = value.state.max;
        const count = value.state.cnt;
        const d = [
          groupid,
          authorid,
          padid,
          tabid,
          sessop,
          timestamp,
          top,
          bottom,
          mintop,
          maxbottom,
          count,
          taskid,
          userid,
          moderator,
        ];
        await DB.query(
          "INSERT INTO pad_chat_scrolling (groupid, authorid, padid, tabid, sessop, timestamp, top, bottom, mintop, maxbottom, count, taskid, userid, moderator) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)",
          d
        );
        this.logs.chat_scrolling++;
      }
    } catch (error) {
      console.error("handleTracking");
      console.error(error);
    }
  }

  public async handleComments(key: string, value: any) {
    try {
      const todo: any = [];
      for (const commentid in value) {
        const comment = value[commentid];
        const authorid =
          typeof comment.author === "string" &&
          comment.author.indexOf("a.") !== -1
            ? comment.author
            : null;
        const text = typeof comment.text === "string" ? comment.text : "";
        const timestamp = new Date(comment.timestamp);
        const userid =
          authorid !== null && this.authors[authorid]
            ? this.authors[authorid]
            : null;
        const moderator = userid !== null ? this._modIds.has(userid) : false;
        let authorname = typeof comment.name === "string" ? comment.name : "";
        const groupid = parseInt(key.split("_")[3]);
        const taskid = parseInt(key.split("_")[1]);
        const padid = key.split(":")[1];
        if (authorname.length > 50) {
          authorname = "";
        }
        const d: any = [
          groupid,
          padid,
          text,
          commentid,
          authorid,
          authorname,
          taskid,
          timestamp,
          userid,
          moderator,
        ];
        todo.push(
          DB.query(
            "INSERT INTO pad_comment (groupid, padid, text, commentid, authorid, authorname, taskid, timestamp, userid, moderator) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)",
            d
          )
        );
        this.logs.comments++;
      }
      await Promise.all(todo);
    } catch (error) {
      console.error("handleComments");
      console.error(error);
    }
  }

  public async handleCommentReplies(key: string, value: any) {
    try {
      for (const replyid in value) {
        const reply = value[replyid];
        const authorid =
          typeof reply.author === "string" && reply.author.indexOf("a.") !== -1
            ? reply.author
            : null;
        const text = typeof reply.text === "string" ? reply.text : "";
        const padid = key.split(":")[1];
        const authorname = reply.author;
        const timestamp = new Date(reply.timestamp);
        const groupid = parseInt(key.split("_")[3]);
        const taskid = parseInt(key.split("_")[1]);
        const commentid = reply.commentId;
        const userid =
          authorid !== null && this.authors[authorid]
            ? this.authors[authorid]
            : null;
        const moderator = userid !== null ? this._modIds.has(userid) : false;
        const d = [
          authorid,
          text,
          padid,
          authorname,
          timestamp,
          groupid,
          replyid,
          taskid,
          commentid,
          userid,
          moderator,
        ];
        await DB.query(
          "INSERT INTO pad_comment_reply (authorid, text, padid, authorname, timestamp, groupid, replyid, taskid, commentid, userid, moderator) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)",
          d
        );
        this.logs.comment_replies++;
      }
    } catch (error) {
      console.error("handleCommentReplies");
      console.error(error);
    }
  }

  public async handlePadInfo(key: string, value: any) {
    try {
      if (key.indexOf("g.") === -1) {
        return;
      }
      const padid = key.split(":")[1];
      const readonlyid = typeof this.readonlyIds[padid]
        ? this.readonlyIds[padid]
        : null;
      const groupid = parseInt(padid.split("_")[3]);
      const lastversion = value;
      const taskid = parseInt(padid.split("_")[1]);
      const d: any = [
        padid,
        readonlyid,
        groupid,
        JSON.stringify(lastversion),
        taskid,
      ];
      await DB.query(
        "INSERT INTO pad (padid, readonlyid, groupid, lastversion, taskid) VALUES ($1, $2, $3, $4, $5)",
        d
      );
      this.logs.padinfo++;
    } catch (error) {
      console.error("handlePadInfo");
      console.error(error);
    }
  }

  public async getAuthors(): Promise<void> {
    for (const entry of this.data) {
      if (entry[0].indexOf("mapper2author") === -1) {
        continue;
      }
      const moodleId = parseInt(entry[0].split(":")[1]);
      const epId = entry[1];
      this.authors[epId] = moodleId;
    }
  }

  public async getReadonlyPadIds(): Promise<void> {
    for (const entry of this.data) {
      if (entry[0].indexOf("pad2readonly") !== -1) {
        const pid = entry[0].split(":")[1];
        this.readonlyIds[pid] = entry[1];
      } else if (entry[0].indexOf("readonly2pad") !== -1) {
        const pid = entry[0].split(":")[1];
        this.readonlyToPads[pid] = entry[1];
      }
    }
  }
}

interface IReadonlyPad {
  [key: string]: string;
}

interface IAuthor {
  [key: string]: number;
}
