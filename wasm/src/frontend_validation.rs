pub(crate) trait FrontendValidation<O> {
    fn frontend_validation(&self, min: O, max: O) -> anyhow::Result<O>;
}

impl FrontendValidation<u32> for Option<i32> {
    fn frontend_validation(&self, min: u32, max: u32) -> anyhow::Result<u32> {
        anyhow::ensure!(
            self.is_some() && self.unwrap() as u32 >= min && self.unwrap() as u32 <= max,
            format!(
                "Value {:?} sould be between {min} and {max}",
                self
            )
        );
        Ok(self.unwrap() as u32)
    }
}

impl FrontendValidation<f64> for Option<f64> {
    fn frontend_validation(&self, min: f64, max: f64) -> anyhow::Result<f64> {
        anyhow::ensure!(
            self.is_some() && self.unwrap() >= min && self.unwrap() <= max,
            format!(
                "Value {:?} sould be between {min} and {max}",
                self
            )
        );
        Ok(self.unwrap())
    }
}