CREATE TABLE IF NOT EXISTS employees (
  id   BIGSERIAL PRIMARY KEY,
  full_name  TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS work_results (
  id                BIGSERIAL PRIMARY KEY,
  recorded_at       TIMESTAMPTZ,
  processed_at      TIMESTAMPTZ DEFAULT NOW(),
  employee_id       BIGINT NOT NULL REFERENCES employees(id) ON DELETE RESTRICT,
  politeness_score  SMALLINT NOT NULL CHECK (politeness_score BETWEEN 0 AND 10),
  problem_solved    BOOLEAN NOT NULL,
  new_record_created BOOLEAN NOT NULL,
  comment           TEXT
);

CREATE INDEX IF NOT EXISTS idx_work_results_employee_time
  ON work_results(employee_id, recorded_at DESC);

CREATE INDEX IF NOT EXISTS idx_work_results_recorded_at
  ON work_results(recorded_at DESC);

CREATE OR REPLACE VIEW employee_stats AS
SELECT
  e.id AS employee_id,
  e.full_name,
  COUNT(w.id)                                     AS records_count,
  AVG(w.politeness_score)::numeric(5,2)           AS avg_politeness,
  AVG((w.problem_solved)::int)::numeric(5,4)      AS solved_rate,
  SUM((w.problem_solved)::int)                    AS total_solved,
  AVG((w.new_record_created)::int)::numeric(5,4)  AS new_record_rate,
  SUM((w.new_record_created)::int)                AS total_new_records
FROM employees e
LEFT JOIN work_results w
  ON w.employee_id = e.id
GROUP BY e.id, e.full_name;

-- Функция для получения employee_id по full_name
CREATE OR REPLACE FUNCTION get_employee_id(p_full_name TEXT)
RETURNS BIGINT AS $$
DECLARE
  v_employee_id BIGINT;
BEGIN
  -- Пытаемся найти существующего сотрудника
  SELECT id INTO v_employee_id FROM employees WHERE full_name = p_full_name;
  
  RETURN v_employee_id;
END;
$$ LANGUAGE plpgsql;

-- Функция для сохранения результата анализа
CREATE OR REPLACE FUNCTION save_work_result(
  p_full_name TEXT,
  p_recorded_at TIMESTAMPTZ,
  p_politeness_score SMALLINT,
  p_problem_solved BOOLEAN,
  p_new_record_created BOOLEAN,
  p_comment TEXT
)
RETURNS BIGINT AS $$
DECLARE
  v_employee_id BIGINT;
  v_result_id BIGINT;
BEGIN
  -- Получаем сотрудника
  v_employee_id := get_employee_id(p_full_name);
  
  -- Если сотрудник не найден, возвращаем NULL
  IF v_employee_id IS NULL THEN
    RAISE NOTICE 'Employee not found: %', p_full_name;
    RETURN NULL;
  END IF;
  
  -- Вставляем результат работы
  INSERT INTO work_results (
    employee_id,
    recorded_at,
    politeness_score,
    problem_solved,
    new_record_created,
    comment
  ) VALUES (
    v_employee_id,
    p_recorded_at,
    p_politeness_score,
    p_problem_solved,
    p_new_record_created,
    p_comment
  )
  RETURNING id INTO v_result_id;
  
  RETURN v_result_id;
END;
$$ LANGUAGE plpgsql;

-- Функция для добавления тестовых данных
CREATE OR REPLACE FUNCTION add_mock_data()
RETURNS VOID AS $$
DECLARE
  v_employee_id BIGINT;
BEGIN
  -- Добавляем тестового сотрудника напрямую
  INSERT INTO employees (full_name) VALUES ('Иванов Иван Иванович')
  ON CONFLICT (full_name) DO NOTHING;
  
  -- Получаем id сотрудника
  v_employee_id := get_employee_id('Иванов Иван Иванович');
  
  -- Добавляем несколько тестовых записей
  INSERT INTO work_results (employee_id, recorded_at, politeness_score, problem_solved, new_record_created, comment)
  VALUES 
    (v_employee_id, NOW() - INTERVAL '2 days', 8, true, true, 'Отлично справился с записью пациента'),
    (v_employee_id, NOW() - INTERVAL '1 day', 9, true, false, 'Вежливо ответил на вопросы'),
    (v_employee_id, NOW(), 7, false, false, 'Клиент отказался от записи');
  
  RAISE NOTICE 'Mock data added for employee: Иванов Иван Иванович';
END;
$$ LANGUAGE plpgsql;

-- Вызываем функцию добавления тестовых данных
SELECT add_mock_data();