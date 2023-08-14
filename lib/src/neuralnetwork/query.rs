#[derive(Debug)]
pub struct QueryEntry {
    pub input: Vec<f64>,
}

#[derive(Debug)]
pub struct QueryData<'a>(pub &'a Vec<QueryEntry>);

#[derive(Debug)]
pub struct QueryResult(pub Vec<f64>);

#[derive(Debug)]
pub struct QueryResults(pub Vec<QueryResult>);