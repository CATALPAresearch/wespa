
CREATE TABLE "pad_visibility" (
  "id" TEXT,
  "groupid" TEXT,
  "authorid" TEXT,
  "padid" TEXT,
  "tabid" TEXT,
  "sessop" TEXT,
  "timestamp" TEXT,
  "state" TEXT,
  "taskid" TEXT,
  "userid" TEXT,
  "moderator" TEXT
);

CREATE TABLE "pad_session" (
  "id" TEXT,
  "groupid" TEXT,
  "authorid" TEXT,
  "padid" TEXT,
  "tabid" TEXT,
  "sessop" TEXT,
  "timestamp" TEXT,
  "state" TEXT,
  "userid" TEXT,
  "taskid" TEXT,
  "moderator" TEXT
);

CREATE TABLE "pad_scrolling" (
  "id" TEXT,
  "groupid" TEXT,
  "authorid" TEXT,
  "padid" TEXT,
  "tabid" TEXT,
  "sessop" TEXT,
  "timestamp" TEXT,
  "top" TEXT,
  "bottom" TEXT,
  "taskid" TEXT,
  "userid" TEXT,
  "count" TEXT,
  "moderator" TEXT
);

CREATE TABLE "pad_review_answer" (
  "id" TEXT,
  "taskid" TEXT,
  "userid" TEXT,
  "groupid" TEXT,
  "answer" TEXT,
  "created" TEXT,
  "modified" TEXT,
  "key" TEXT
);

CREATE TABLE "pad_chat_scrolling" (
  "id" TEXT,
  "groupid" TEXT,
  "authorid" TEXT,
  "padid" TEXT,
  "tabid" TEXT,
  "sessop" TEXT,
  "timestamp" TEXT,
  "top" TEXT,
  "bottom" TEXT,
  "mintop" TEXT,
  "maxbottom" TEXT,
  "count" TEXT,
  "taskid" TEXT,
  "userid" TEXT,
  "moderator" TEXT
);


CREATE TABLE "pad" (
  "id" TEXT,
  "padid" TEXT,
  "readonlyid" TEXT,
  "groupid" TEXT,
  "lastversion" TEXT,
  "taskid" TEXT
);

CREATE TABLE "task" (
  "id" TEXT,
  "semester" TEXT,
  "description" TEXT,
  "courseid" TEXT
);

CREATE TABLE "pad_comment" (
  "id" TEXT,
  "groupid" TEXT,
  "padid" TEXT,
  "text" TEXT,
  "commentid" TEXT,
  "authorid" TEXT,
  "authorname" TEXT,
  "taskid" TEXT,
  "timestamp" TEXT,
  "userid" TEXT,
  "moderator" TEXT
);

CREATE TABLE "pad_chat" (
  "id" TEXT,
  "groupid" TEXT,
  "rev" TEXT,
  "text" TEXT,
  "padid" TEXT,
  "authorid" TEXT,
  "timestamp" TEXT,
  "taskid" TEXT,
  "userid" TEXT,
  "moderator" TEXT
);

CREATE TABLE "pad_review_groupassignment" (
  "id" TEXT,
  "groupid" TEXT,
  "peergroupid" TEXT,
  "taskid" TEXT
);

CREATE TABLE "pad_chat_visibility" (
  "id" TEXT,
  "groupid" TEXT,
  "authorid" TEXT,
  "padid" TEXT,
  "tabid" TEXT,
  "sessop" TEXT,
  "timestamp" TEXT,
  "state" TEXT,
  "taskid" TEXT,
  "userid" TEXT,
  "moderator" TEXT
);

CREATE TABLE "pad_commit" (
  "id" TEXT,
  "groupid" TEXT,
  "padid" TEXT,
  "rev" TEXT,
  "text" TEXT,
  "authorid" TEXT,
  "timestamp" TEXT,
  "taskid" TEXT,
  "userid" TEXT,
  "moderator" TEXT,
  "textprevlen" TEXT,
  "textnewlen" TEXT,
  "textnumcharsadd" TEXT,
  "textnumcharsdel" TEXT
);

CREATE TABLE "pad_comment_reply" (
  "id" TEXT,
  "authorid" TEXT,
  "text" TEXT,
  "padid" TEXT,
  "authorname" TEXT,
  "timestamp" TEXT,
  "groupid" TEXT,
  "replyid" TEXT,
  "taskid" TEXT,
  "commentid" TEXT,
  "userid" TEXT,
  "moderator" TEXT
);

CREATE TABLE "pad_review_question" (
  "id" TEXT,
  "taskid" TEXT,
  "heading" TEXT,
  "key" TEXT,
  "question" TEXT,
  "max" TEXT,
  "min" TEXT,
  "created" TEXT,
  "modified" TEXT
);