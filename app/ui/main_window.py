from __future__ import annotations

import csv
from pathlib import Path
from datetime import datetime

from PySide6.QtCore import Qt, QThread
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSpinBox,
    QDoubleSpinBox,
    QSplitter,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.core.config_manager import AppConfig, ConfigManager
from app.core.prompt_builder import build_user_prompt
from app.services.openai_compatible_client import OpenAICompatibleClient
from app.workers.connection_worker import ConnectionWorker
from app.workers.stream_worker import StreamWorker


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('OpenAI Compatible Coding Assistant')
        self.resize(1600, 920)

        self.config_manager = ConfigManager()
        self.config = self.config_manager.load()
        self.session_started_at = datetime.now()
        self.log_path = self._create_session_log()

        self.stream_thread: QThread | None = None
        self.connection_thread: QThread | None = None

        self._build_ui()
        self._apply_config_to_ui(self.config)
        self._load_default_template()
        self._log(f'Session started. CSV log: {self.log_path}')

    def _build_ui(self) -> None:
        self._build_menu_bar()

        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        workspace_splitter = QSplitter(Qt.Orientation.Horizontal)
        workspace_splitter.addWidget(self._build_left_panel())
        workspace_splitter.addWidget(self._build_right_panel())
        workspace_splitter.setChildrenCollapsible(False)
        workspace_splitter.setStretchFactor(0, 1)
        workspace_splitter.setStretchFactor(1, 1)
        root.addWidget(workspace_splitter)

        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage('Ready')

    def _build_menu_bar(self) -> None:
        self._build_connection_dialog()
        self._build_import_dialog()

        menu_bar = self.menuBar()

        connection_menu = menu_bar.addMenu('Connection / Config')
        self.open_connection_action = connection_menu.addAction('Open Connection / Config')
        self.open_connection_action.triggered.connect(self._show_connection_dialog)
        connection_menu.addSeparator()
        self.test_connection_action = connection_menu.addAction('Test Connection')
        self.test_connection_action.triggered.connect(self._on_test_connection)
        self.save_config_action = connection_menu.addAction('Save YAML')
        self.save_config_action.triggered.connect(self._on_save_yaml)
        self.reload_config_action = connection_menu.addAction('Reload YAML')
        self.reload_config_action.triggered.connect(self._on_reload_yaml)

        import_menu = menu_bar.addMenu('Import Layer')
        self.open_import_action = import_menu.addAction('Open Import Layer')
        self.open_import_action.triggered.connect(self._show_import_dialog)
        self.load_import_action = import_menu.addAction('Load File to Import Layer')
        self.load_import_action.triggered.connect(self._on_load_import_file)
        self.clear_import_action = import_menu.addAction('Clear Import Layer')
        self.clear_import_action.triggered.connect(self.import_edit.clear)

    def _build_connection_dialog(self) -> None:
        self.connection_dialog = QDialog(self)
        self.connection_dialog.setWindowTitle('Connection / Config')
        self.connection_dialog.resize(760, 420)
        layout = QVBoxLayout(self.connection_dialog)

        form = QFormLayout()

        self.base_url_edit = QLineEdit()
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.model_name_edit = QLineEdit()
        self.system_prompt_edit = QPlainTextEdit()
        self.system_prompt_edit.setPlaceholderText('System prompt')
        self.system_prompt_edit.setFixedHeight(120)

        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 2.0)
        self.temperature_spin.setSingleStep(0.1)

        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(1, 200000)

        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(5, 3600)

        self.stream_checkbox = QCheckBox('Use output stream type')

        form.addRow('Base URL', self.base_url_edit)
        form.addRow('API Key', self.api_key_edit)
        form.addRow('Model Name', self.model_name_edit)
        form.addRow('Temperature', self.temperature_spin)
        form.addRow('Max Tokens', self.max_tokens_spin)
        form.addRow('Timeout (s)', self.timeout_spin)
        form.addRow('Stream', self.stream_checkbox)
        form.addRow('System Prompt', self.system_prompt_edit)
        layout.addLayout(form)

        button_row = QHBoxLayout()
        self.test_button = QPushButton('Test Connection')
        self.save_config_button = QPushButton('Save YAML')
        self.load_config_button = QPushButton('Reload YAML')
        close_button = QPushButton('Close')

        self.test_button.clicked.connect(self._on_test_connection)
        self.save_config_button.clicked.connect(self._on_save_yaml)
        self.load_config_button.clicked.connect(self._on_reload_yaml)
        close_button.clicked.connect(self.connection_dialog.close)

        button_row.addWidget(self.test_button)
        button_row.addWidget(self.save_config_button)
        button_row.addWidget(self.load_config_button)
        button_row.addStretch(1)
        button_row.addWidget(close_button)
        layout.addLayout(button_row)

    def _build_import_dialog(self) -> None:
        self.import_dialog = QDialog(self)
        self.import_dialog.setWindowTitle('Import Layer')
        self.import_dialog.resize(840, 560)
        layout = QVBoxLayout(self.import_dialog)

        layout.addWidget(QLabel('Project context, notes, references, and summaries for the model.'))

        self.import_edit = QPlainTextEdit()
        self.import_edit.setPlaceholderText(
            'Import Layer: project context, related notes, selected file summaries...'
        )
        layout.addWidget(self.import_edit)

        button_row = QHBoxLayout()
        load_button = QPushButton('Load File')
        clear_button = QPushButton('Clear')
        close_button = QPushButton('Close')

        load_button.clicked.connect(self._on_load_import_file)
        clear_button.clicked.connect(self.import_edit.clear)
        close_button.clicked.connect(self.import_dialog.close)

        button_row.addWidget(load_button)
        button_row.addWidget(clear_button)
        button_row.addStretch(1)
        button_row.addWidget(close_button)
        layout.addLayout(button_row)

    def _build_left_panel(self) -> QWidget:
        container = QWidget(self)
        layout = QVBoxLayout(container)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self._build_current_code_group())
        splitter.addWidget(self._build_issue_group())
        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter)

        return container

    def _build_right_panel(self) -> QWidget:
        container = QWidget(self)
        layout = QVBoxLayout(container)

        action_row = QHBoxLayout()
        self.run_button = QPushButton('Run')
        self.clear_output_button = QPushButton('Clear Output')
        self.copy_input_button = QPushButton('Copy Request Input')

        self.run_button.clicked.connect(self._on_run)
        self.clear_output_button.clicked.connect(self._clear_output)
        self.copy_input_button.clicked.connect(self._on_copy_request_preview)

        action_row.addWidget(self.run_button)
        action_row.addWidget(self.clear_output_button)
        action_row.addWidget(self.copy_input_button)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self._build_request_preview_group())
        splitter.addWidget(self._build_output_group())
        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)
        layout.addWidget(splitter)

        return container

    def _build_current_code_group(self) -> QGroupBox:
        group = QGroupBox('Current Code')
        layout = QVBoxLayout(group)

        self.current_code_edit = QPlainTextEdit()
        self.current_code_edit.setPlaceholderText('Current code')
        layout.addWidget(self.current_code_edit)

        button_row = QHBoxLayout()
        self.load_current_code_button = QPushButton('Load File')
        clear_button = QPushButton('Clear')
        self.load_current_code_button.clicked.connect(self._on_load_current_code_file)
        clear_button.clicked.connect(self.current_code_edit.clear)
        button_row.addWidget(self.load_current_code_button)
        button_row.addWidget(clear_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        return group

    def _build_issue_group(self) -> QGroupBox:
        group = QGroupBox('Bug / Issue')
        layout = QVBoxLayout(group)

        self.issue_edit = QPlainTextEdit()
        self.issue_edit.setPlaceholderText('Bug code / issue / traceback / expected behavior')
        layout.addWidget(self.issue_edit)

        button_row = QHBoxLayout()
        self.load_issue_button = QPushButton('Load File')
        clear_button = QPushButton('Clear')
        self.load_issue_button.clicked.connect(self._on_load_issue_file)
        clear_button.clicked.connect(self.issue_edit.clear)
        button_row.addWidget(self.load_issue_button)
        button_row.addWidget(clear_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        return group

    def _build_request_preview_group(self) -> QGroupBox:
        group = QGroupBox('Displayed Input')
        layout = QVBoxLayout(group)

        self.request_preview_edit = QPlainTextEdit()
        self.request_preview_edit.setReadOnly(True)
        self.request_preview_edit.setPlaceholderText('Input preview sent to model')
        layout.addWidget(self.request_preview_edit)

        return group

    def _build_output_group(self) -> QGroupBox:
        group = QGroupBox('Modified Code / Model Output')
        layout = QVBoxLayout(group)

        self.output_edit = QTextEdit()
        self.output_edit.setReadOnly(True)
        layout.addWidget(self.output_edit)

        return group

    def _load_default_template(self) -> None:
        self.import_edit.setPlainText(
            'This is a quick PySide6 initialization-order test.\n'
            'Please inspect the widget creation order and fix the startup crash.'
        )
        self.current_code_edit.setPlainText(
            'from PySide6.QtWidgets import QPushButton, QTextEdit\n'
            '\n'
            '\n'
            'class DemoWindow:\n'
            '    def __init__(self):\n'
            '        self.build_ui()\n'
            '\n'
            '    def build_ui(self):\n'
            "        self.clear_button = QPushButton('Clear Output')\n"
            '        self.clear_button.clicked.connect(self.output_edit.clear)\n'
            '        self.output_edit = QTextEdit()\n'
        )
        self.issue_edit.setPlainText(
            'App startup raises AttributeError because a widget is referenced before it is created.\n'
            'Reorder the initialization so the signal connection is safe.'
        )

    def _show_connection_dialog(self) -> None:
        self.connection_dialog.show()
        self.connection_dialog.raise_()
        self.connection_dialog.activateWindow()

    def _show_import_dialog(self) -> None:
        self.import_dialog.show()
        self.import_dialog.raise_()
        self.import_dialog.activateWindow()

    def _apply_config_to_ui(self, config: AppConfig) -> None:
        self.base_url_edit.setText(config.base_url)
        self.api_key_edit.setText(config.api_key)
        self.model_name_edit.setText(config.model_name)
        self.system_prompt_edit.setPlainText(config.system_prompt)
        self.temperature_spin.setValue(config.temperature)
        self.max_tokens_spin.setValue(config.max_tokens)
        self.timeout_spin.setValue(config.timeout_seconds)
        self.stream_checkbox.setChecked(config.stream)
        self._log(f'Config loaded from: {self.config_manager.config_path}')

    def _read_config_from_ui(self) -> AppConfig:
        return AppConfig(
            base_url=self.base_url_edit.text().strip(),
            api_key=self.api_key_edit.text().strip(),
            model_name=self.model_name_edit.text().strip(),
            system_prompt=self.system_prompt_edit.toPlainText().strip(),
            temperature=float(self.temperature_spin.value()),
            max_tokens=int(self.max_tokens_spin.value()),
            stream=self.stream_checkbox.isChecked(),
            timeout_seconds=int(self.timeout_spin.value()),
        )

    def _build_client(self) -> OpenAICompatibleClient:
        config = self._read_config_from_ui()
        return OpenAICompatibleClient(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout_seconds=config.timeout_seconds,
        )

    def _validate_before_request(self) -> bool:
        if not self.base_url_edit.text().strip():
            QMessageBox.warning(self, 'Missing Base URL', 'Please enter Base URL.')
            self._show_connection_dialog()
            return False
        if not self.model_name_edit.text().strip():
            QMessageBox.warning(self, 'Missing Model Name', 'Please enter Model Name.')
            self._show_connection_dialog()
            return False
        return True

    def _set_busy(self, busy: bool) -> None:
        self.run_button.setEnabled(not busy)
        self.test_button.setEnabled(not busy)
        self.save_config_button.setEnabled(not busy)
        self.load_config_button.setEnabled(not busy)
        self.test_connection_action.setEnabled(not busy)
        self.save_config_action.setEnabled(not busy)
        self.reload_config_action.setEnabled(not busy)
        self.statusBar().showMessage('Working...' if busy else 'Ready')

    def _on_load_import_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, 'Open Import Layer File')
        if path:
            self.import_edit.setPlainText(Path(path).read_text(encoding='utf-8', errors='ignore'))
            self._log(f'Loaded import layer file: {path}')

    def _on_load_current_code_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, 'Open Code File')
        if path:
            self.current_code_edit.setPlainText(Path(path).read_text(encoding='utf-8', errors='ignore'))
            self._log(f'Loaded current code file: {path}')

    def _on_load_issue_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, 'Open Issue File')
        if path:
            self.issue_edit.setPlainText(Path(path).read_text(encoding='utf-8', errors='ignore'))
            self._log(f'Loaded issue file: {path}')

    def _on_save_yaml(self) -> None:
        config = self._read_config_from_ui()
        self.config_manager.save(config)
        self._log(f'YAML saved: {self.config_manager.config_path}')
        QMessageBox.information(self, 'Saved', f'Config saved to\n{self.config_manager.config_path}')

    def _on_reload_yaml(self) -> None:
        self.config = self.config_manager.load()
        self._apply_config_to_ui(self.config)

    def _on_copy_request_preview(self) -> None:
        self.request_preview_edit.selectAll()
        self.request_preview_edit.copy()
        self._log('Request input copied to clipboard.')

    def _clear_output(self) -> None:
        self.output_edit.clear()
        self._log('Model output cleared.')

    def _on_test_connection(self) -> None:
        if not self._validate_before_request():
            return
        self._set_busy(True)
        client = self._build_client()
        model_name = self.model_name_edit.text().strip()

        self.connection_thread = QThread(self)
        self.connection_worker = ConnectionWorker(client, model_name)
        self.connection_worker.moveToThread(self.connection_thread)
        self.connection_thread.started.connect(self.connection_worker.run)
        self.connection_worker.result_signal.connect(self._on_connection_result)
        self.connection_worker.finished_signal.connect(self.connection_thread.quit)
        self.connection_worker.finished_signal.connect(self.connection_worker.deleteLater)
        self.connection_thread.finished.connect(self.connection_thread.deleteLater)
        self.connection_thread.finished.connect(lambda: self._set_busy(False))
        self.connection_thread.start()
        self._log('Testing connection...')

    def _on_connection_result(self, ok: bool, message: str) -> None:
        self._log(message)
        title = 'Connection OK' if ok else 'Connection Failed'
        if ok:
            QMessageBox.information(self, title, message)
        else:
            QMessageBox.warning(self, title, message)

    def _on_run(self) -> None:
        if not self._validate_before_request():
            return

        config = self._read_config_from_ui()
        prompt = build_user_prompt(
            import_context=self.import_edit.toPlainText(),
            current_code=self.current_code_edit.toPlainText(),
            bug_report=self.issue_edit.toPlainText(),
        )
        self.request_preview_edit.setPlainText(prompt)
        self.output_edit.clear()
        self._log('Run requested.')
        self._set_busy(True)

        client = self._build_client()
        self.stream_thread = QThread(self)
        self.stream_worker = StreamWorker(
            client=client,
            model_name=config.model_name,
            system_prompt=config.system_prompt,
            user_prompt=prompt,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            use_stream=config.stream,
        )
        self.stream_worker.moveToThread(self.stream_thread)
        self.stream_thread.started.connect(self.stream_worker.run)
        self.stream_worker.chunk_received.connect(self._append_output_chunk)
        self.stream_worker.started_signal.connect(lambda: self._log('Streaming started.'))
        self.stream_worker.error_signal.connect(self._on_stream_error)
        self.stream_worker.finished_signal.connect(self.stream_thread.quit)
        self.stream_worker.finished_signal.connect(self.stream_worker.deleteLater)
        self.stream_thread.finished.connect(self.stream_thread.deleteLater)
        self.stream_thread.finished.connect(lambda: self._set_busy(False))
        self.stream_thread.start()

    def _append_output_chunk(self, text: str) -> None:
        cursor = self.output_edit.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.output_edit.setTextCursor(cursor)
        self.output_edit.ensureCursorVisible()

    def _on_stream_error(self, message: str) -> None:
        self._log(f'Error: {message}')
        QMessageBox.warning(self, 'Run Error', message)

    def _create_session_log(self) -> Path:
        log_dir = Path(__file__).resolve().parents[2] / 'log'
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = self.session_started_at.strftime('%Y%m%d_%H%M%S')
        log_path = log_dir / f'{timestamp}.csv'
        with log_path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['timestamp', 'message'])
        return log_path

    def _log(self, message: str) -> None:
        timestamp = datetime.now().isoformat(timespec='seconds')
        with self.log_path.open('a', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow([timestamp, message])
        if self.statusBar() is not None:
            self.statusBar().showMessage(message, 5000)
