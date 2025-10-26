import React, { useState, useEffect } from 'react';
import { Form, Input, Button, Table, Space, Select, message, Upload } from 'antd';
import { PlusOutlined, DeleteOutlined, UploadOutlined } from '@ant-design/icons';

const { Item: FormItem } = Form;
const { Option } = Select;

/**
 * Volatility Input Component
 * Allows users to input volatility surface data for pricing
 */
const VolatilityInput = ({ value = [], onChange, disabled = false }) => {
  const [volatilitySurface, setVolatilitySurface] = useState(value);
  const [inputMode, setInputMode] = useState('manual'); // 'manual' or 'file'

  useEffect(() => {
    setVolatilitySurface(value);
  }, [value]);

  const handleAddPoint = () => {
    const newPoint = {
      strike: '',
      expiry: '',
      volatility: '',
      key: Date.now() // Unique key for table
    };
    const newSurface = [...volatilitySurface, newPoint];
    setVolatilitySurface(newSurface);
    onChange?.(newSurface);
  };

  const handleRemovePoint = (key) => {
    const newSurface = volatilitySurface.filter(item => item.key !== key);
    setVolatilitySurface(newSurface);
    onChange?.(newSurface);
  };

  const handlePointChange = (key, field, newValue) => {
    const newSurface = volatilitySurface.map(item => {
      if (item.key === key) {
        return { ...item, [field]: newValue };
      }
      return item;
    });
    setVolatilitySurface(newSurface);
    onChange?.(newSurface);
  };

  const validateStrike = (strike) => {
    const numStrike = parseFloat(strike);
    return !isNaN(numStrike) && numStrike > 0 && numStrike <= 3.0; // 30% to 300% moneyness
  };

  const validateExpiry = (expiry) => {
    const numExpiry = parseFloat(expiry);
    return !isNaN(numExpiry) && numExpiry > 0 && numExpiry <= 10; // Up to 10 years
  };

  const validateVolatility = (volatility) => {
    const numVol = parseFloat(volatility);
    return !isNaN(numVol) && numVol > 0 && numVol <= 2.0; // 0% to 200% vol
  };

  const loadDefaultSurface = () => {
    const defaultSurface = [
      // ATM volatilities
      { strike: 1.0, expiry: 0.25, volatility: 0.25, key: 1 },
      { strike: 1.0, expiry: 0.5, volatility: 0.24, key: 2 },
      { strike: 1.0, expiry: 1.0, volatility: 0.23, key: 3 },
      { strike: 1.0, expiry: 2.0, volatility: 0.22, key: 4 },
      { strike: 1.0, expiry: 5.0, volatility: 0.21, key: 5 },

      // OTM calls (strike < 1.0)
      { strike: 0.9, expiry: 1.0, volatility: 0.26, key: 6 },
      { strike: 0.8, expiry: 1.0, volatility: 0.28, key: 7 },

      // OTM puts (strike > 1.0)
      { strike: 1.1, expiry: 1.0, volatility: 0.26, key: 8 },
      { strike: 1.2, expiry: 1.0, volatility: 0.29, key: 9 }
    ];
    setVolatilitySurface(defaultSurface);
    onChange?.(defaultSurface);
    message.success('Default volatility surface loaded');
  };

  const clearSurface = () => {
    setVolatilitySurface([]);
    onChange?.([]);
  };

  const handleFileUpload = (file) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const csvText = e.target.result;
        const lines = csvText.split('\n').filter(line => line.trim());
        const headers = lines[0].split(',').map(h => h.trim().toLowerCase());

        // Expected format: strike,expiry,volatility
        const strikeIndex = headers.indexOf('strike');
        const expiryIndex = headers.indexOf('expiry');
        const volIndex = headers.indexOf('volatility');

        if (strikeIndex === -1 || expiryIndex === -1 || volIndex === -1) {
          throw new Error('CSV must contain columns: strike, expiry, volatility');
        }

        const surface = [];
        for (let i = 1; i < lines.length; i++) {
          const values = lines[i].split(',').map(v => v.trim());
          if (values.length >= 3) {
            surface.push({
              strike: parseFloat(values[strikeIndex]),
              expiry: parseFloat(values[expiryIndex]),
              volatility: parseFloat(values[volIndex]),
              key: Date.now() + i
            });
          }
        }

        setVolatilitySurface(surface);
        onChange?.(surface);
        message.success(`Loaded ${surface.length} volatility points from file`);
      } catch (error) {
        message.error(`Failed to parse CSV file: ${error.message}`);
      }
    };
    reader.readAsText(file);
    return false; // Prevent default upload behavior
  };

  const exportToCSV = () => {
    if (volatilitySurface.length === 0) {
      message.warning('No data to export');
      return;
    }

    const csvContent = [
      'strike,expiry,volatility',
      ...volatilitySurface.map(point => `${point.strike},${point.expiry},${point.volatility}`)
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', 'volatility_surface.csv');
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const columns = [
    {
      title: 'Strike (Moneyness)',
      dataIndex: 'strike',
      key: 'strike',
      render: (text, record) => (
        <Input
          value={text}
          onChange={(e) => handlePointChange(record.key, 'strike', e.target.value)}
          placeholder="e.g., 1.0"
          disabled={disabled}
          style={{ width: '100px' }}
        />
      ),
      width: 140
    },
    {
      title: 'Expiry (Years)',
      dataIndex: 'expiry',
      key: 'expiry',
      render: (text, record) => (
        <Input
          value={text}
          onChange={(e) => handlePointChange(record.key, 'expiry', e.target.value)}
          placeholder="e.g., 1.0"
          disabled={disabled}
          style={{ width: '100px' }}
        />
      ),
      width: 120
    },
    {
      title: 'Volatility',
      dataIndex: 'volatility',
      key: 'volatility',
      render: (text, record) => (
        <Input
          value={text}
          onChange={(e) => handlePointChange(record.key, 'volatility', e.target.value)}
          placeholder="e.g., 0.25"
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
          onClick={() => handleRemovePoint(record.key)}
          disabled={disabled}
        >
          Remove
        </Button>
      ),
      width: 100
    }
  ];

  const isValidSurface = volatilitySurface.every(item =>
    validateStrike(item.strike) && validateExpiry(item.expiry) && validateVolatility(item.volatility)
  );

  return (
    <div>
      <Space style={{ marginBottom: 16 }}>
        <Select value={inputMode} onChange={setInputMode} disabled={disabled}>
          <Option value="manual">Manual Input</Option>
          <Option value="file">File Upload</Option>
        </Select>

        {inputMode === 'manual' && (
          <>
            <Button
              type="default"
              icon={<PlusOutlined />}
              onClick={handleAddPoint}
              disabled={disabled}
            >
              Add Point
            </Button>
            <Button
              type="default"
              onClick={loadDefaultSurface}
              disabled={disabled}
            >
              Load Default Surface
            </Button>
          </>
        )}

        <Button
          type="default"
          danger
          onClick={clearSurface}
          disabled={disabled}
        >
          Clear All
        </Button>

        <Button
          type="default"
          onClick={exportToCSV}
          disabled={disabled || volatilitySurface.length === 0}
        >
          Export CSV
        </Button>
      </Space>

      {inputMode === 'file' && (
        <div style={{ marginBottom: 16 }}>
          <Upload
            accept=".csv"
            beforeUpload={handleFileUpload}
            showUploadList={false}
            disabled={disabled}
          >
            <Button icon={<UploadOutlined />} disabled={disabled}>
              Upload CSV File
            </Button>
          </Upload>
          <div style={{ marginTop: 8, fontSize: '12px', color: '#666' }}>
            CSV format: strike,expiry,volatility (one point per row)
          </div>
        </div>
      )}

      <Table
        columns={columns}
        dataSource={volatilitySurface}
        pagination={false}
        size="small"
        rowKey="key"
        scroll={{ y: 300 }}
        locale={{ emptyText: inputMode === 'manual'
          ? 'No volatility surface data. Click "Add Point" to begin.'
          : 'Upload a CSV file to load volatility surface data.'
        }}
      />

      {volatilitySurface.length > 0 && (
        <div style={{ marginTop: 16 }}>
          <Space>
            <span>Surface Points: {volatilitySurface.length}</span>
            {isValidSurface ? (
              <span style={{ color: 'green' }}>✓ Valid surface</span>
            ) : (
              <span style={{ color: 'red' }}>✗ Invalid surface data</span>
            )}
          </Space>
        </div>
      )}

      {!isValidSurface && volatilitySurface.length > 0 && (
        <div style={{ marginTop: 8, color: 'red', fontSize: '12px' }}>
          Please ensure strikes are between 0.3-3.0, expiries are between 0.1-10 years, and volatilities are between 0.01-2.0.
        </div>
      )}
    </div>
  );
};

export default VolatilityInput;