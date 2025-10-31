#![no_std]
extern crate alloc;

use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};
use core::{
    fmt::{self, Display},
    result::Result,
};

/// The [`TransactionLogs`] object.
///
/// Its internal representation is a list of [`InstructionCall`], an object that
/// represents a single call to a specific instruction and holds both its logs and
/// potentially other nested objects of [`InstructionCall`].
#[derive(Clone, Debug)]
pub struct TransactionLogs {
    pub invocations: Vec<InstructionCall>,
}

impl TransactionLogs {
    /// Get all the program logs contained in the parsed output
    pub fn logs(&self) -> impl Iterator<Item = &String> {
        self.invocations.iter().flat_map(move |i| i.logs(true))
    }

    /// Get all the program logs for a specific program address in the parsed output
    pub fn program_logs(&self, program_id: &str) -> impl Iterator<Item = &String> {
        let invocations = self
            .invocations
            .iter()
            .flat_map(|i| i.children(true).chain(core::iter::once(i)));

        invocations
            .filter_map(move |i| {
                if i.program_id == program_id {
                    Some(i.logs(false))
                } else {
                    None
                }
            })
            .flatten()
    }
}

/// Inner events nested in a [`InstructionCall`] object
#[derive(Clone, Debug)]
pub enum Event {
    /// A log
    Log(String),
    /// A nested instruction invocation
    Invoke(InstructionCall),
}

#[derive(Clone, Debug)]
/// Represents a single instruction invocation and its output
pub struct InstructionCall {
    /// The program id
    pub program_id: String,
    /// Sequence of logs and child invocations in order
    pub events: Vec<Event>,
    /// `Program _ consumed _ of _ compute units`
    pub consumed_line: Option<String>,
    /// `Program return: _`
    pub program_return_line: Option<String>,
    /// `Program _ failed: _` | `Program _ success`
    pub program_result_line: Option<String>,
}

/// Error variants returned when parsing logs
#[derive(Debug)]
pub enum LogParseError {
    /// Failed parsing specific log line
    InvalidLogLine(String),
    /// Instruction index is out of bounds
    InvalidInvokeIndex(String),
}

impl Display for LogParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LogParseError::InvalidLogLine(log) => write!(f, "Invalid log: {}", log),
            LogParseError::InvalidInvokeIndex(log) => write!(f, "Invalid invoke index: {}", log),
        }
    }
}

impl core::error::Error for LogParseError {}

impl InstructionCall {
    /// Parses CU information from the parsed logs.
    pub fn cu_stats(&self) -> Option<(u32, u32)> {
        let cu_log = self
            .consumed_line
            .as_ref()?
            .split_whitespace()
            .collect::<Vec<_>>();
        let consumed = cu_log.get(3)?.parse::<u32>().ok()?;
        let out_of = cu_log.get(5)?.parse::<u32>().ok()?;
        Some((consumed, out_of))
    }

    /// Parses the program execution status from the parsed logs.
    ///
    /// * Returns `Ok(())` if program executed successfully
    /// * Returns `Err(String)` containing the execution error if the program failed
    pub fn program_status(&self) -> Option<Result<(), String>> {
        let line = self.program_result_line.as_ref()?;
        let rest = line
            .strip_prefix(&format!("Program {}", self.program_id))
            .map(|x| x.trim())?;
        if rest.starts_with("success") {
            Some(Ok(()))
        } else {
            let error = rest.strip_prefix("failed:")?.trim();
            Some(Err(error.to_string()))
        }
    }

    /// Parses the program return data from the parsed logs.
    pub fn program_return_data(&self) -> Option<String> {
        let return_data = self
            .program_return_line
            .as_ref()?
            .strip_prefix(&format!("Program return: {}", self.program_id))?
            .trim();
        Some(return_data.to_string())
    }

    /// Returns the program logs emitted by this instruction.
    ///
    /// If the `nested` argument is true, include program logs emitted by inner CPI
    /// calls made by this instruction.
    pub fn logs<'a>(&'a self, nested: bool) -> impl Iterator<Item = &'a String> + 'a {
        let mut vec = Vec::new();
        for event in &self.events {
            match event {
                Event::Log(log) => vec.push(log),
                Event::Invoke(child) => {
                    if nested {
                        vec.extend(child.logs(true));
                    }
                }
            }
        }
        vec.into_iter()
    }

    /// Returns the child [`InstructionCall`] objects for this instruction.
    ///
    /// If the `nested` argument is true, include calls not made directly by this
    /// instruction but still exist in its tree of execution at a lower level.
    pub fn children(&self, nested: bool) -> impl Iterator<Item = &InstructionCall> {
        let mut vec = Vec::new();
        for event in &self.events {
            if let Event::Invoke(child) = event {
                vec.push(child);
                if nested {
                    vec.extend(child.children(true))
                }
            }
        }
        vec.into_iter()
    }
}

impl TransactionLogs {
    /// Parse transaction logs into a list of instructions, each containing logs and potentially
    /// nested calls to other inner instructions.
    pub fn new(logs: Vec<String>) -> Result<Self, LogParseError> {
        // estimate 6 top-level instructions in a transaction
        let mut invocations = Vec::<InstructionCall>::with_capacity(6);
        // max cpi call-depth is 4
        let mut stack = Vec::<(usize, InstructionCall)>::with_capacity(4);

        for line in logs {
            let words = line.split_whitespace().collect::<Vec<_>>();
            let Some(second) = words.get(1) else {
                return Err(LogParseError::InvalidLogLine(line));
            };

            if *second == "log:" || *second == "data:" {
                if let Some((_, invocation)) = stack.last_mut() {
                    invocation.events.push(Event::Log(line));
                }
                continue;
            } else if *second == "return:" {
                if let Some((_, invocation)) = stack.last_mut() {
                    invocation.program_return_line = Some(line);
                }
                continue;
            }

            if let Some(third) = words.get(2) {
                if *third == "invoke" {
                    // we have a new invocation. `second` is the program-id
                    let invocation = InstructionCall {
                        program_id: second.to_string(),
                        events: Vec::with_capacity(8),
                        consumed_line: None,
                        program_return_line: None,
                        program_result_line: None,
                    };

                    let level = words
                        .get(3)
                        .and_then(|s| {
                            s.chars()
                                .take(2)
                                .last()
                                .and_then(|s| s.to_digit(10).and_then(|d| usize::try_from(d).ok()))
                        })
                        .ok_or(LogParseError::InvalidInvokeIndex(line))?;

                    while stack.last().is_some_and(|(l, _)| *l >= level) {
                        if let Some((_, completed)) = stack.pop() {
                            if let Some((parent_level, parent)) = stack.last_mut() {
                                if *parent_level < level {
                                    parent.events.push(Event::Invoke(completed));
                                }
                            } else {
                                invocations.push(completed);
                            }
                        }
                    }
                    stack.push((level, invocation));
                } else if *third == "success" || *third == "failed" {
                    let program_id = *second;
                    let matches = stack
                        .last()
                        .is_some_and(|(_, invocation)| invocation.program_id == program_id);

                    if matches {
                        if let Some((_, invocation)) = stack.last_mut() {
                            invocation.program_result_line = Some(line);
                        }
                        if let Some((level, completed)) = stack.pop() {
                            if let Some((parent_level, parent)) = stack.last_mut() {
                                if *parent_level < level {
                                    parent.events.push(Event::Invoke(completed));
                                }
                            } else {
                                invocations.push(completed);
                            }
                        }
                    }
                } else if *third == "consumed" {
                    if let Some((_, invocation)) = stack.last_mut() {
                        invocation.consumed_line = Some(line);
                    }
                }
            }
        }

        while let Some((_, invocation)) = stack.pop() {
            invocations.push(invocation);
        }

        Ok(TransactionLogs { invocations })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn create_test_logs() -> Vec<String> {
        vec![
            "Program ComputeBudget111111111111111111111111111111 invoke [1]".to_string(),
            "Program ComputeBudget111111111111111111111111111111 success".to_string(),
            "Program ComputeBudget111111111111111111111111111111 invoke [1]".to_string(),
            "Program ComputeBudget111111111111111111111111111111 success".to_string(),
            "Program ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL invoke [1]".to_string(),
            "Program log: CreateIdempotent".to_string(),
            "Program TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA invoke [2]".to_string(),
            "Program log: Instruction: GetAccountDataSize".to_string(),
            "Program TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA consumed 1569 of 68230 compute units".to_string(),
            "Program return: TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA pQAAAAAAAAA=".to_string(),
            "Program TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA success".to_string(),
            "Program 11111111111111111111111111111111 invoke [2]".to_string(),
            "Program 11111111111111111111111111111111 success".to_string(),
            "Program log: Initialize the associated token account".to_string(),
            "Program TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA invoke [2]".to_string(),
            "Program log: Instruction: InitializeImmutableOwner".to_string(),
            "Program log: Please upgrade to SPL Token 2022 for immutable owner support".to_string(),
            "Program TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA consumed 1405 of 61643 compute units".to_string(),
            "Program TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA success".to_string(),
            "Program TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA invoke [2]".to_string(),
            "Program log: Instruction: InitializeAccount3".to_string(),
            "Program TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA consumed 3158 of 57761 compute units".to_string(),
            "Program TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA success".to_string(),
            "Program ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL consumed 19315 of 73635 compute units".to_string(),
            "Program ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL success".to_string(),
            "Program 11111111111111111111111111111111 invoke [1]".to_string(),
            "Program 11111111111111111111111111111111 success".to_string(),
            "Program TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA invoke [1]".to_string(),
            "Program log: Instruction: SyncNative".to_string(),
            "Program TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA consumed 3045 of 54170 compute units".to_string(),
            "Program TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA success".to_string(),
            "Program JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4 invoke [1]".to_string(),
            "Program log: Instruction: Route".to_string(),
            "Program 675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8 invoke [2]".to_string(),
            "Program log: ray_log: A+Ev0QAAAAAAAAAAAAAAAAABAAAAAAAAAOEv0QAAAAAABcaYQYoFAACqoy+MfwEAAMuTAwMAAAAA".to_string(),
            "Program TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA invoke [3]".to_string(),
            "Program log: Instruction: Transfer".to_string(),
            "Program TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA consumed 4736 of 32513 compute units".to_string(),
            "Program TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA success".to_string(),
            "Program TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA invoke [3]".to_string(),
            "Program log: Instruction: Transfer".to_string(),
            "Program TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA consumed 4645 of 25308 compute units".to_string(),
            "Program TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA success".to_string(),
            "Program 675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8 consumed 26627 of 46596 compute units".to_string(),
            "Program 675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8 success".to_string(),
            "Program JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4 invoke [2]".to_string(),
            "Program JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4 consumed 195 of 18472 compute units".to_string(),
            "Program JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4 success".to_string(),
            "Program JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4 consumed 33938 of 51125 compute units".to_string(),
            "Program return: JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4 y5MDAwAAAAA=".to_string(),
            "Program JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4 success".to_string(),
            "Program TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA invoke [1]".to_string(),
            "Program log: Instruction: CloseAccount".to_string(),
            "Program TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA consumed 2915 of 17187 compute units".to_string(),
            "Program TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA success".to_string(),
        ]
    }

    #[test]
    fn test_parse_transaction_logs() {
        let logs = create_test_logs();
        let transaction_logs = TransactionLogs::new(logs).expect("Failed to parse logs");

        assert_eq!(
            transaction_logs.invocations.len(),
            7,
            "Expected 7 top-level invocations"
        );

        let program_ids: Vec<&str> = transaction_logs
            .invocations
            .iter()
            .map(|inv| inv.program_id.as_str())
            .collect();
        assert_eq!(
            program_ids,
            vec![
                "ComputeBudget111111111111111111111111111111",
                "ComputeBudget111111111111111111111111111111",
                "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL",
                "11111111111111111111111111111111",
                "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
                "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4",
                "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
            ],
            "Unexpected program IDs"
        );
    }

    #[test]
    fn test_get_cu_consumption() {
        let logs = create_test_logs();
        let transaction_logs = TransactionLogs::new(logs).expect("Failed to parse logs");

        // Test compute unit consumption for specific programs
        let atoken_invocation = transaction_logs
            .invocations
            .iter()
            .find(|inv| inv.program_id == "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")
            .expect("AToken invocation not found");
        assert_eq!(
            atoken_invocation.cu_stats(),
            Some((19315, 73635)),
            "Unexpected CU consumption for AToken"
        );

        let tokenkeg_invocation = transaction_logs
            .invocations
            .iter()
            .find(|inv| inv.program_id == "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
            .expect("Tokenkeg invocation not found");
        assert_eq!(
            tokenkeg_invocation.cu_stats(),
            Some((3045, 54170)),
            "Unexpected CU consumption for Tokenkeg"
        );

        let jup_invocation = transaction_logs
            .invocations
            .iter()
            .find(|inv| inv.program_id == "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4")
            .expect("JUP invocation not found");
        assert_eq!(
            jup_invocation.cu_stats(),
            Some((33938, 51125)),
            "Unexpected CU consumption for JUP"
        );
    }

    #[test]
    fn test_get_program_status() {
        let logs = create_test_logs();
        let transaction_logs = TransactionLogs::new(logs).expect("Failed to parse logs");

        let atoken_invocation = transaction_logs
            .invocations
            .iter()
            .find(|inv| inv.program_id == "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")
            .expect("AToken invocation not found");
        assert_eq!(
            atoken_invocation.program_status(),
            Some(Ok(())),
            "Expected AToken success"
        );

        let jup_invocation = transaction_logs
            .invocations
            .iter()
            .find(|inv| inv.program_id == "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4")
            .expect("JUP invocation not found");
        assert_eq!(
            jup_invocation.program_status(),
            Some(Ok(())),
            "Expected JUP success"
        );
    }

    #[test]
    fn test_get_program_return() {
        let logs = create_test_logs();
        let transaction_logs = TransactionLogs::new(logs).expect("Failed to parse logs");

        let tokenkeg_invocation = transaction_logs
            .invocations
            .iter()
            .find(|inv| inv.program_id == "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
            .expect("Tokenkeg invocation not found");
        assert_eq!(
            tokenkeg_invocation.program_return_data(),
            None,
            "Expected no return data for Tokenkeg"
        );

        let jup_invocation = transaction_logs
            .invocations
            .iter()
            .find(|inv| inv.program_id == "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4")
            .expect("JUP invocation not found");
        assert_eq!(
            jup_invocation.program_return_data(),
            Some("y5MDAwAAAAA=".to_string()),
            "Unexpected return data for JUP"
        );
    }

    #[test]
    fn test_program_logs_include_inner() {
        let logs = create_test_logs();
        let transaction_logs = TransactionLogs::new(logs).expect("Failed to parse logs");

        // Test logs for AToken including inner logs
        let atoken_invocation = transaction_logs
            .invocations
            .iter()
            .find(|inv| inv.program_id == "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")
            .expect("AToken invocation not found");
        let logs: Vec<&String> = atoken_invocation.logs(true).collect();
        assert_eq!(
            logs,
            vec![
                &"Program log: CreateIdempotent".to_string(),
                &"Program log: Instruction: GetAccountDataSize".to_string(),
                &"Program log: Initialize the associated token account".to_string(),
                &"Program log: Instruction: InitializeImmutableOwner".to_string(),
                &"Program log: Please upgrade to SPL Token 2022 for immutable owner support"
                    .to_string(),
                &"Program log: Instruction: InitializeAccount3".to_string(),
            ],
            "Unexpected logs for AToken with inner"
        );
    }

    #[test]
    fn test_program_logs_exclude_inner() {
        let logs = create_test_logs();
        let transaction_logs = TransactionLogs::new(logs).expect("Failed to parse logs");

        // Test logs for AToken excluding inner logs
        let atoken_invocation = transaction_logs
            .invocations
            .iter()
            .find(|inv| inv.program_id == "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")
            .expect("AToken invocation not found");
        let logs: Vec<&String> = atoken_invocation.logs(false).collect();
        assert_eq!(
            logs,
            vec![
                &"Program log: CreateIdempotent".to_string(),
                &"Program log: Initialize the associated token account".to_string(),
            ],
            "Unexpected logs for AToken without inner"
        );
    }

    #[test]
    fn test_nested_invocations() {
        let logs = create_test_logs();
        let transaction_logs = TransactionLogs::new(logs).expect("Failed to parse logs");

        let jup_invocation = transaction_logs
            .invocations
            .iter()
            .find(|inv| inv.program_id == "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4")
            .expect("JUP invocation not found");
        assert_eq!(
            jup_invocation.children(false).count(),
            2,
            "Expected 2 child invocations for JUP"
        );

        let child_program_ids: Vec<&str> = jup_invocation
            .children(false)
            .map(|inv| inv.program_id.as_str())
            .collect();
        assert_eq!(
            child_program_ids,
            vec![
                "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",
                "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4"
            ],
            "Unexpected child program IDs for JUP"
        );

        let child_675_invocation = jup_invocation
            .children(false)
            .find(|inv| inv.program_id == "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8")
            .expect("675kPX invocation not found");
        assert_eq!(
            child_675_invocation.children(false).count(),
            2,
            "Expected 2 child invocations for 675kPX"
        );
        let grandchild_program_ids: Vec<&str> = child_675_invocation
            .children(false)
            .map(|inv| inv.program_id.as_str())
            .collect();
        assert_eq!(
            grandchild_program_ids,
            vec![
                "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
                "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
            ],
            "Unexpected grandchild program IDs for 675kPX"
        );
    }

    #[test]
    fn test_invalid_log_line() {
        let mut logs = create_test_logs();
        logs.push("Invalid".to_string());
        let result = TransactionLogs::new(logs);
        assert!(matches!(result, Err(LogParseError::InvalidLogLine(_))));
    }

    #[test]
    fn test_invalid_invoke_index() {
        let mut logs = create_test_logs();
        logs.insert(
            0,
            "Program ComputeBudget111111111111111111111111111111 invoke [invalid]".to_string(),
        );
        let result = TransactionLogs::new(logs);
        assert!(matches!(result, Err(LogParseError::InvalidInvokeIndex(_))));
    }
}
