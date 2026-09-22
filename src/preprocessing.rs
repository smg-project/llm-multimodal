/// Request-specific inputs for multimodal preprocessing.
#[derive(Debug, Clone, Copy, Default)]
pub struct PreprocessingContext {
    /// Tokens available for this preprocessing call's media after fixed prompt tokens
    /// have been accounted for, including any model-specific structural overhead.
    /// The caller resolves context-length limits and allocates the budget across modalities.
    /// `None` means no budget was supplied; `Some(0)` means the budget is exhausted.
    pub token_budget: Option<usize>,
}
