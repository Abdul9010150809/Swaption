import React, { useState, useEffect } from 'react';
import { Form, Input, Button, Table, Space, message } from 'antd';
import { PlusOutlined, DeleteOutlined } from '@ant-design/icons';

const { Item: FormItem } = Form;

/**
 * Yield Curve Input Component
 * Allows users to input or modify yield curve data for pricing
 */
const YieldCurveInput = ({ value = [], onChange, disabled = false }) => {
  const [yieldCurve, setYieldCurve] = useState(value);

  useEffect(() => {
    setYieldCurve(value);
  }, [value]);

  const handleAddTenor = () => {
    const newTenor = {
      tenor: '',
      rate: '',
      key: Date.now() // Unique key for table
    };
    const newCurve = [...yieldCurve, newTenor];
    setYieldCurve(newCurve);
    onChange?.(newCurve);
  };

  const handleRemoveTenor = (key) => {
    const newCurve = yieldCurve.filter(item => item.key !== key);
    setYieldCurve(newCurve);
    onChange?.(newCurve);
  };

  const handleTenorChange = (key, field, newValue) => {
    const newCurve = yieldCurve.map(item => {
      if (item.key === key) {
        return { ...item, [field]: newValue };
      }
      return item;
    });
    setYieldCurve(newCurve);
    onChange?.(newCurve);
  };

  const validateTenor = (tenor) => {
    const numTenor = parseFloat(tenor);
    return !isNaN(numTenor) && numTenor > 0 && numTenor <= 50;
  };

  const validateRate = (rate) => {
    const numRate = parseFloat(rate);
    return !isNaN(numRate) && numRate >= -0.01 && numRate <= 0.20;
  };

  const loadDefaultCurve = () => {
    const defaultCurve = [
      { tenor: 0.25, rate: 0.035, key: 1 },
      { tenor: 0.5, rate: 0.037, key: 2 },
      { tenor: 1, rate: 0.039, key: 3 },
      { tenor: 2, rate: 0.041, key: 4 },
      { tenor: 5, rate: 0.043, key: 5 },
      { tenor: 10, rate: 0.045, key: 6 },
      { tenor: 20, rate: 0.047, key: 7 },
      { tenor: 30, rate: 0.048, key: 8 }
    ];
    setYieldCurve(defaultCurve);
    onChange?.(defaultCurve);
    message.success('Default yield curve loaded');
  };

  const clearCurve = () => {
    setYieldCurve([]);
    onChange?.([]);
  };

  const columns = [
    {
      title: 'Tenor (Years)',
      dataIndex: 'tenor',
      key: 'tenor',
      render: (text, record) => (
        <Input
          value={text}
          onChange={(e) => handleTenorChange(record.key, 'tenor', e.target.value)}
          placeholder="e.g., 1.0"
          disabled={disabled}
          style={{ width: '100px' }}
        />
      ),
      width: 120
    },
    {
      title: 'Rate (%)',
      dataIndex: 'rate',
      key: 'rate',
      render: (text, record) => (
        <Input
          value={text}
          onChange={(e) => handleTenorChange(record.key, 'rate', e.target.value)}
          placeholder="e.g., 3.50"
          disabled={disabled}
          style={{ width: '100px' }}
        />
      ),
      width: 120
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, record) => (
        <Button
          type="text"
          danger
          icon={<DeleteOutlined />}
          onClick={() => handleRemoveTenor(record.key)}
          disabled={disabled}
        >
          Remove
        </Button>
      ),
      width: 100
    }
  ];

  const isValidCurve = yieldCurve.every(item =>
    validateTenor(item.tenor) && validateRate(item.rate)
  );

  return (
    <div>
      <Space style={{ marginBottom: 16 }}>
        <Button
          type="default"
          icon={<PlusOutlined />}
          onClick={handleAddTenor}
          disabled={disabled}
        >
          Add Tenor
        </Button>
        <Button
          type="default"
          onClick={loadDefaultCurve}
          disabled={disabled}
        >
          Load Default Curve
        </Button>
        <Button
          type="default"
          danger
          onClick={clearCurve}
          disabled={disabled}
        >
          Clear All
        </Button>
      </Space>

      <Table
        columns={columns}
        dataSource={yieldCurve}
        pagination={false}
        size="small"
        rowKey="key"
        locale={{ emptyText: 'No yield curve data. Click "Add Tenor" to begin.' }}
      />

      {yieldCurve.length > 0 && (
        <div style={{ marginTop: 16 }}>
          <Space>
            <span>Curve Points: {yieldCurve.length}</span>
            {isValidCurve ? (
              <span style={{ color: 'green' }}>✓ Valid curve</span>
            ) : (
              <span style={{ color: 'red' }}>✗ Invalid curve data</span>
            )}
          </Space>
        </div>
      )}

      {!isValidCurve && yieldCurve.length > 0 && (
        <div style={{ marginTop: 8, color: 'red', fontSize: '12px' }}>
          Please ensure all tenors are between 0.1-50 years and rates are between -1% and 20%.
        </div>
      )}
    </div>
  );
};

export default YieldCurveInput;