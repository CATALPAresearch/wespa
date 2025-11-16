import pg from "pg";

export default class DB {
  private static readonly _config: pg.ClientConfig = Object.freeze({
    host: "localhost",
    port: 5432,
    user: "postgres",
    password: "postgres",
  });

  private static readonly _pool: pg.Pool = new pg.Pool(
    Object.assign({}, this._config, { database: "cw_analytics" })
  );

  public static query(
    sql: string,
    params: any[] = []
  ): Promise<pg.QueryResult<any>> {
    return new Promise((resolve, reject) => {
      DB._pool.query(sql, params, (err, res) => {
        if (err) {
          return reject(err);
        }
        return resolve(res);
      });
    });
  }
}
